import type {
  AgentRunRead,
  FeedbackRead,
  FeedbackVerdict,
  LlmOption,
  PlannerRequest,
  TokenResponse,
  UserRead,
} from '../types'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ??
  'http://localhost:8000'

type RequestOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  headers?: Record<string, string>
}

class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? 'GET',
    // Auth is an httpOnly cookie set by POST /auth/login, not a token this
    // client ever sees - 'include' is required for the browser to send it
    // cross-origin (frontend/backend are different ports) and to accept
    // the Set-Cookie response on login.
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail =
      typeof payload?.detail === 'string'
        ? payload.detail
        : Array.isArray(payload?.detail)
          ? payload.detail.map((item: { msg?: string }) => item.msg).join(', ')
          : `Request failed with status ${response.status}`
    throw new ApiError(detail, response.status)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function signup(input: {
  email: string
  password: string
  full_name: string
}): Promise<UserRead> {
  return request<UserRead>('/auth/signup', {
    method: 'POST',
    body: input,
  })
}

export async function login(input: {
  email: string
  password: string
}): Promise<TokenResponse> {
  return request<TokenResponse>('/auth/login', {
    method: 'POST',
    body: input,
  })
}

export async function logout(): Promise<void> {
  return request<void>('/auth/logout', { method: 'POST' })
}

export async function fetchCurrentUser(): Promise<UserRead> {
  return request<UserRead>('/auth/me')
}

export async function createAgentRun(
  payload: PlannerRequest,
  apiKey?: string,
): Promise<AgentRunRead> {
  return request<AgentRunRead>('/agent-runs', {
    method: 'POST',
    body: payload,
    headers: apiKey ? { 'X-LLM-API-Key': apiKey } : undefined,
  })
}

export async function fetchLlmOptions(): Promise<LlmOption[]> {
  return request<LlmOption[]>('/llm-options')
}

export async function submitFeedback(payload: {
  recommendation_id: number
  session_uuid: string
  verdict: FeedbackVerdict
}): Promise<FeedbackRead> {
  return request<FeedbackRead>('/feedback', {
    method: 'POST',
    body: payload,
  })
}

export { ApiError, API_BASE_URL }
