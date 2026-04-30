import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import './App.css'
import {
  ApiError,
  API_BASE_URL,
  createAgentRun,
  fetchCurrentUser,
  login,
  signup,
} from './lib/api'
import type { AgentRunRead, AuthMode, SessionState } from './types'

function App() {
  const [authMode, setAuthMode] = useState<AuthMode>('login')
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

  useEffect(() => {
    const raw = window.localStorage.getItem('smart-travel-session')
    if (!raw) {
      return
    }

    try {
      const parsed = JSON.parse(raw) as SessionState
      setSession(parsed)
    } catch {
      window.localStorage.removeItem('smart-travel-session')
    }
  }, [])

  function persistSession(nextSession: SessionState | null) {
    setSession(nextSession)

    if (nextSession) {
      window.localStorage.setItem(
        'smart-travel-session',
        JSON.stringify(nextSession),
      )
      return
    }

    window.localStorage.removeItem('smart-travel-session')
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

  function handleLogout() {
    persistSession(null)
    setResult(null)
  }

  return (
    <main className="shell">
      <section className="hero-panel">
        <p className="eyebrow">Smart Travel Assistant</p>
        <h1>Prompt-first trip planning with your backend agent in the loop.</h1>
        <p className="hero-copy">
          This MVP signs in, sends one natural-language travel request to
          LangGraph, and shows the saved recommendation with its tool trail.
        </p>
        <div className="hero-meta">
          <span>API: {API_BASE_URL}</span>
          <span>Delivery: Discord webhook enabled in backend</span>
        </div>
      </section>

      <section className="workspace">
        <div className="panel auth-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">Authentication</p>
              <h2>{session ? 'Signed in' : 'Connect to the backend'}</h2>
            </div>
            {!session ? (
              <div
                className="segmented-control"
                role="tablist"
                aria-label="Auth mode"
              >
                <button
                  type="button"
                  className={authMode === 'login' ? 'active' : ''}
                  onClick={() => setAuthMode('login')}
                >
                  Login
                </button>
                <button
                  type="button"
                  className={authMode === 'signup' ? 'active' : ''}
                  onClick={() => setAuthMode('signup')}
                >
                  Sign up
                </button>
              </div>
            ) : null}
          </div>

          {session ? (
            <div className="session-card">
              <p className="session-name">{session.user.full_name || 'Traveler'}</p>
              <p>{session.user.email}</p>
              <p className="session-meta">
                User #{session.user.id} - Active {session.user.is_active ? 'yes' : 'no'}
              </p>
              <button
                type="button"
                className="secondary-button"
                onClick={handleLogout}
              >
                Log out
              </button>
            </div>
          ) : (
            <form className="form-grid" onSubmit={handleAuthSubmit}>
              {authMode === 'signup' ? (
                <label>
                  <span>Full name</span>
                  <input
                    value={fullName}
                    onChange={(event) => setFullName(event.target.value)}
                    placeholder="Kayan"
                  />
                </label>
              ) : null}
              <label>
                <span>Email</span>
                <input
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="you@example.com"
                  required
                />
              </label>
              <label>
                <span>Password</span>
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="At least 8 characters"
                  required
                />
              </label>
              {authError ? <p className="error-text">{authError}</p> : null}
              <button type="submit" className="primary-button" disabled={authPending}>
                {authPending
                  ? 'Working...'
                  : authMode === 'login'
                    ? 'Login and load session'
                    : 'Create account and continue'}
              </button>
            </form>
          )}
        </div>

        <div className="panel planner-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">Planner</p>
              <h2>Ask for a trip recommendation</h2>
            </div>
            <span className="status-pill">{session ? 'Ready' : 'Auth required'}</span>
          </div>

          <form className="form-grid" onSubmit={handlePlanSubmit}>
            <label>
              <span>Prompt</span>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={7}
                required
              />
            </label>
            <label className="compact-field">
              <span>RAG top K</span>
              <input
                type="number"
                min={1}
                max={8}
                value={retrievalTopK}
                onChange={(event) => setRetrievalTopK(Number(event.target.value))}
              />
            </label>
            {plannerError ? <p className="error-text">{plannerError}</p> : null}
            <button
              type="submit"
              className="primary-button"
              disabled={plannerPending || !session}
            >
              {plannerPending ? 'Planning trip...' : 'Run agent'}
            </button>
          </form>
        </div>
      </section>

      <section className="results-grid">
        <article className="panel result-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">Final answer</p>
              <h2>Saved recommendation</h2>
            </div>
            <span className={`status-pill status-${result?.status || 'idle'}`}>
              {result?.status || 'No run yet'}
            </span>
          </div>

          {result ? (
            <>
              <div className="result-meta">
                <span>Run #{result.id}</span>
                <span>{new Date(result.created_at).toLocaleString()}</span>
              </div>
              <p className="prompt-preview">{result.prompt}</p>
              <div className="response-card">
                {result.response.split('\n').map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
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

        <article className="panel logs-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">Tool trail</p>
              <h2>What the agent used</h2>
            </div>
            <span className="status-pill">
              {result ? `${result.tool_logs.length} logs` : 'No logs yet'}
            </span>
          </div>

          {result?.tool_logs.length ? (
            <div className="logs-list">
              {result.tool_logs.map((log) => (
                <article key={log.id} className="log-card">
                  <div className="log-header">
                    <strong>{log.tool_name}</strong>
                    <span className={`log-status status-${log.status}`}>{log.status}</span>
                  </div>
                  <p className="log-time">
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
              the classifier, recommender, RAG, weather, Claude, and Discord
              delivery path.
            </p>
          )}
        </article>
      </section>
    </main>
  )
}

export default App
