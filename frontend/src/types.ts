export type AuthMode = 'login' | 'signup'

export interface UserRead {
  id: number
  email: string
  full_name: string | null
  is_active: boolean
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface PlannerRequest {
  prompt: string
  retrieval_top_k: number
}

export interface ToolLogRead {
  id: number
  agent_run_id: number
  tool_name: string
  input_payload: string
  output_payload: string
  status: string
  created_at: string
}

export interface AgentRunRead {
  id: number
  user_id: number
  prompt: string
  response: string
  status: string
  created_at: string
  tool_logs: ToolLogRead[]
}

export interface SessionState {
  token: string
  user: UserRead
}
