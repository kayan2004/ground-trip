import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import './App.css'
import groundtripLogo from './assets/groundtrip-logo.svg'
import {
  ApiError,
  createAgentRun,
  fetchCurrentUser,
  login,
  signup,
  submitFeedback,
} from './lib/api'
import type { AgentRunRead, AuthMode, FeedbackVerdict, SessionState } from './types'
import { WhyThisPick } from './WhyThisPick'

type View = 'login' | 'signup' | 'app'

const APP_ROUTE = '/app'
const LOGIN_ROUTE = '/login'
const SIGNUP_ROUTE = '/signup'
const FEEDBACK_SESSION_STORAGE_KEY = 'smart-travel-feedback-session-uuid'

function getOrCreateFeedbackSessionUuid(): string {
  const existing = window.localStorage.getItem(FEEDBACK_SESSION_STORAGE_KEY)
  if (existing) {
    return existing
  }
  const created = crypto.randomUUID()
  window.localStorage.setItem(FEEDBACK_SESSION_STORAGE_KEY, created)
  return created
}

function getViewFromPath(pathname: string): View {
  if (pathname === SIGNUP_ROUTE) {
    return 'signup'
  }
  if (pathname === APP_ROUTE) {
    return 'app'
  }
  return 'login'
}

function navigateTo(view: View, replace = false) {
  const target =
    view === 'signup' ? SIGNUP_ROUTE : view === 'app' ? APP_ROUTE : LOGIN_ROUTE
  const nextUrl = `${target}${window.location.search}${window.location.hash}`

  if (replace) {
    window.history.replaceState(null, '', nextUrl)
    return
  }

  window.history.pushState(null, '', nextUrl)
}

function statusPillTone(status: string | undefined): string {
  if (status === 'completed') return 'gt-pill--positive'
  if (status === 'partial') return 'gt-pill--brass'
  if (status === 'failed') return 'gt-pill--negative'
  return ''
}

function renderInlineBoldText(text: string) {
  const parts = text.split(/(\*\*.*?\*\*)/g)

  return parts.map((part, index) => {
    const isBold = part.startsWith('**') && part.endsWith('**') && part.length > 4
    if (!isBold) {
      return <span key={`${part}-${index}`}>{part}</span>
    }

    return (
      <strong key={`${part}-${index}`} className="markdown-strong">
        {part.slice(2, -2)}
      </strong>
    )
  })
}

function App() {
  const [view, setView] = useState<View>(() => getViewFromPath(window.location.pathname))
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [prompt, setPrompt] = useState(
    'I have two weeks off in July and around $1,500. I want somewhere warm, not too touristy, and I like hiking. Where should I go, when should I book, and what should I expect?',
  )
  const [retrievalTopK, setRetrievalTopK] = useState(3)
  const [session, setSession] = useState<SessionState | null>(null)
  const [result, setResult] = useState<AgentRunRead | null>(null)
  const [authError, setAuthError] = useState('')
  const [plannerError, setPlannerError] = useState('')
  const [authPending, setAuthPending] = useState(false)
  const [plannerPending, setPlannerPending] = useState(false)
  const [feedbackSessionUuid] = useState(getOrCreateFeedbackSessionUuid)
  const [feedbackByRecommendation, setFeedbackByRecommendation] = useState<
    Record<number, FeedbackVerdict>
  >({})
  const [feedbackError, setFeedbackError] = useState('')

  const authMode: AuthMode = view === 'signup' ? 'signup' : 'login'

  useEffect(() => {
    const handlePopState = () => {
      setView(getViewFromPath(window.location.pathname))
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    const raw = window.localStorage.getItem('smart-travel-session')
    if (!raw) {
      if (view === 'app') {
        navigateTo('login', true)
        setView('login')
      }
      return
    }

    try {
      const parsed = JSON.parse(raw) as SessionState
      setSession(parsed)

      if (getViewFromPath(window.location.pathname) !== 'app') {
        navigateTo('app', true)
        setView('app')
      }
    } catch {
      window.localStorage.removeItem('smart-travel-session')
      if (view === 'app') {
        navigateTo('login', true)
        setView('login')
      }
    }
  }, [])

  function persistSession(nextSession: SessionState | null) {
    setSession(nextSession)

    if (nextSession) {
      window.localStorage.setItem(
        'smart-travel-session',
        JSON.stringify(nextSession),
      )
      navigateTo('app')
      setView('app')
      return
    }

    window.localStorage.removeItem('smart-travel-session')
    navigateTo('login')
    setView('login')
  }

  function handleAuthViewChange(nextMode: AuthMode) {
    setAuthError('')
    const nextView = nextMode === 'signup' ? 'signup' : 'login'
    navigateTo(nextView)
    setView(nextView)
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setAuthPending(true)
    setAuthError('')

    try {
      if (authMode === 'signup') {
        await signup({
          email,
          password,
          full_name: fullName.trim(),
        })
      }

      const token = await login({ email, password })
      const user = await fetchCurrentUser(token.access_token)
      persistSession({ token: token.access_token, user })
    } catch (error) {
      setAuthError(
        error instanceof ApiError ? error.message : 'Authentication failed.',
      )
    } finally {
      setAuthPending(false)
    }
  }

  async function handlePlanSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!session) {
      setPlannerError('Please log in first.')
      navigateTo('login')
      setView('login')
      return
    }

    setPlannerPending(true)
    setPlannerError('')

    try {
      const agentRun = await createAgentRun(session.token, {
        prompt,
        retrieval_top_k: retrievalTopK,
      })
      setResult(agentRun)
    } catch (error) {
      setPlannerError(
        error instanceof ApiError ? error.message : 'Trip planning failed.',
      )
    } finally {
      setPlannerPending(false)
    }
  }

  async function handleFeedback(recommendationId: number, verdict: FeedbackVerdict) {
    setFeedbackError('')

    try {
      await submitFeedback({
        recommendation_id: recommendationId,
        session_uuid: feedbackSessionUuid,
        verdict,
      })
      setFeedbackByRecommendation((previous) => ({
        ...previous,
        [recommendationId]: verdict,
      }))
    } catch (error) {
      setFeedbackError(
        error instanceof ApiError ? error.message : 'Feedback could not be submitted.',
      )
    }
  }

  function handleLogout() {
    persistSession(null)
    setResult(null)
  }

  if (view !== 'app' || !session) {
    return (
      <main className="gt-auth-grid">
        <section className="gt-panel auth-hero">
          <img src={groundtripLogo} alt="GroundTrip" className="gt-logo" />
          <h1 className="auth-hero-heading">Sign in before you ask the agent where to go.</h1>
          <div className="pipeline-stages">
            <span className="gt-pill gt-pill--brass">extract</span>
            <span className="gt-pill">recommend</span>
            <span className="gt-pill">RAG</span>
            <span className="gt-pill">weather</span>
            <span className="gt-pill gt-pill--positive">synthesize</span>
          </div>
        </section>

        <section className="gt-panel auth-page-panel">
          <div className="gt-panel-header gt-auth-header">
            <div>
              <p className="gt-eyebrow">Authentication</p>
              <h2>{authMode === 'login' ? 'Welcome back' : 'Create your account'}</h2>
            </div>
            <div className="gt-segmented" role="tablist" aria-label="Auth mode">
              <button
                type="button"
                className={authMode === 'login' ? 'gt-segmented-btn active' : 'gt-segmented-btn'}
                onClick={() => handleAuthViewChange('login')}
              >
                Login
              </button>
              <button
                type="button"
                className={authMode === 'signup' ? 'gt-segmented-btn active' : 'gt-segmented-btn'}
                onClick={() => handleAuthViewChange('signup')}
              >
                Sign up
              </button>
            </div>
          </div>

          <form className="form-grid" onSubmit={handleAuthSubmit}>
            {authMode === 'signup' ? (
              <label className="gt-field">
                <span>Full name</span>
                <input
                  className="gt-input"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Kayan"
                  autoComplete="name"
                />
              </label>
            ) : null}
            <label className="gt-field">
              <span>Email</span>
              <input
                className="gt-input"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>
            <label className="gt-field">
              <span>Password</span>
              <input
                className="gt-input"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="At least 8 characters"
                autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                required
              />
            </label>
            {authError ? (
              <p className="error-text" role="alert">
                {authError}
              </p>
            ) : null}
            <button type="submit" className="gt-btn gt-btn--primary" disabled={authPending}>
              {authPending
                ? 'Working…'
                : authMode === 'login'
                  ? 'Login and continue'
                  : 'Create account and continue'}
            </button>
          </form>
        </section>
      </main>
    )
  }

  return (
    <main className="shell">
      <section className="gt-panel hero-panel">
        <div className="gt-planner-header">
          <div>
            <img src={groundtripLogo} alt="GroundTrip" className="gt-logo" />
            <h1 className="planner-heading">Prompt-first trip planning with your backend agent in the loop.</h1>
          </div>
          <div className="gt-panel gt-panel--raised gt-user-card">
            <div className="session-chip">
              <strong>{session.user.full_name || 'Traveler'}</strong>
              <span>{session.user.email}</span>
            </div>
            <button type="button" className="gt-btn gt-btn--ghost" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </div>

        <div className="gt-panel planner-inner">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Planner</p>
              <h2>Ask for a trip recommendation</h2>
            </div>
            <span className="gt-pill gt-pill--positive">ready</span>
          </div>

          <form className="form-grid" onSubmit={handlePlanSubmit}>
            <label className="gt-field">
              <span>Prompt</span>
              <textarea
                className="gt-textarea"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={7}
                required
              />
            </label>
            <label className="gt-field gt-compact-field">
              <span>RAG top K</span>
              <input
                className="gt-input"
                type="number"
                min={1}
                max={8}
                value={retrievalTopK}
                onChange={(event) => setRetrievalTopK(Number(event.target.value))}
              />
            </label>
            {plannerError ? (
              <p className="error-text" role="alert">
                {plannerError}
              </p>
            ) : null}
            <button type="submit" className="gt-btn gt-btn--primary" disabled={plannerPending}>
              {plannerPending ? 'Planning trip…' : 'Run agent'}
            </button>
          </form>
        </div>
      </section>

      <section className="gt-results-grid">
        <article className="gt-panel">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Final answer</p>
              <h2>Saved recommendation</h2>
            </div>
            <span className={`gt-pill ${statusPillTone(result?.status)}`}>
              {result?.status || 'no run yet'}
            </span>
          </div>

          {result ? (
            <>
              <div className="result-meta gt-mono">
                <span>run #{result.id}</span>
                <span>{new Date(result.created_at).toLocaleString()}</span>
              </div>
              <p className="gt-panel gt-panel--paper prompt-preview">{result.prompt}</p>
              <div className="response-card">
                {result.response.split('\n').map((line, index) => (
                  <p key={`${line}-${index}`}>{renderInlineBoldText(line)}</p>
                ))}
              </div>
            </>
          ) : (
            <p className="empty-state">
              Your first successful agent run will show up here with the final
              saved answer from the backend.
            </p>
          )}
        </article>

        <article className="gt-panel">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Recommendations</p>
              <h2>Rate the ranked slate</h2>
            </div>
            <span className="gt-pill">
              {result ? `${result.recommendations.length} destinations` : 'no slate yet'}
            </span>
          </div>

          {feedbackError ? (
            <p className="error-text" role="alert">
              {feedbackError}
            </p>
          ) : null}

          {result?.recommendations.length ? (
            <div className="logs-list">
              {result.recommendations.map((recommendation) => {
                const activeVerdict = feedbackByRecommendation[recommendation.id]
                return (
                  <article key={recommendation.id} className="gt-panel gt-panel--raised log-card">
                    <div className="log-header">
                      <strong>
                        #{recommendation.rank_position} {recommendation.destination_name}, {recommendation.country}
                      </strong>
                      <span className="gt-mono" style={{ color: 'var(--brass)' }}>
                        {recommendation.score.toFixed(4)}
                      </span>
                    </div>
                    <WhyThisPick features={recommendation.features} />
                    <div className="feedback-actions">
                      <button
                        type="button"
                        className={
                          activeVerdict === 1
                            ? 'gt-stamp gt-stamp--positive gt-stamp--active'
                            : 'gt-stamp gt-stamp--positive'
                        }
                        aria-pressed={activeVerdict === 1}
                        onClick={() => handleFeedback(recommendation.id, 1)}
                      >
                        Good match
                      </button>
                      <button
                        type="button"
                        className={
                          activeVerdict === -1
                            ? 'gt-stamp gt-stamp--negative gt-stamp--active'
                            : 'gt-stamp gt-stamp--negative'
                        }
                        aria-pressed={activeVerdict === -1}
                        onClick={() => handleFeedback(recommendation.id, -1)}
                      >
                        Not a fit
                      </button>
                    </div>
                  </article>
                )
              })}
            </div>
          ) : (
            <p className="empty-state">
              Recommended destinations will appear here after a planner run, so
              you can rate each ranked result.
            </p>
          )}
        </article>

        <article className="gt-panel">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Tool trail</p>
              <h2>What the agent used</h2>
            </div>
            <span className="gt-pill">
              {result ? `${result.tool_logs.length} logs` : 'no logs yet'}
            </span>
          </div>

          {result?.tool_logs.length ? (
            <div className="logs-list">
              {result.tool_logs.map((log) => (
                <article key={log.id} className="gt-panel gt-panel--raised log-card">
                  <div className="log-header">
                    <strong>{log.tool_name}</strong>
                    <span className={`gt-pill ${log.status === 'success' ? 'gt-pill--positive' : 'gt-pill--negative'}`}>
                      {log.status}
                    </span>
                  </div>
                  <p className="log-time gt-mono-sm" style={{ color: 'var(--text-tertiary)' }}>
                    {new Date(log.created_at).toLocaleString()}
                  </p>
                  <details>
                    <summary>Input payload</summary>
                    <pre>{log.input_payload}</pre>
                  </details>
                  <details>
                    <summary>Output payload</summary>
                    <pre>{log.output_payload}</pre>
                  </details>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state">
              Tool logs will appear here after a planner run so you can inspect
              the recommender, RAG, weather, LLM synthesis, and Discord
              delivery path.
            </p>
          )}
        </article>
      </section>
    </main>
  )
}

export default App
