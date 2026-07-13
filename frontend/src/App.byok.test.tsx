import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import type { SessionState } from './types'

const sampleSession: SessionState = {
  token: 'jwt-token',
  user: { id: 1, email: 'traveler@test.com', full_name: 'Traveler', is_active: true, created_at: '2026-01-01T00:00:00Z' },
}

const sampleLlmOptions = [
  { provider: 'gemini', model: 'gemini-3.1-flash-lite' },
  { provider: 'openai', model: 'gpt-5.4-nano' },
]

function mockFetch() {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url.includes('/llm-options')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(sampleLlmOptions),
      })
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({}),
    })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  window.localStorage.setItem('smart-travel-session', JSON.stringify(sampleSession))
  window.sessionStorage.clear()
  window.history.pushState(null, '', '/app')
  mockFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

describe('BYOK panel', () => {
  it('is collapsed by default', async () => {
    render(<App />)

    const summary = await screen.findByText('Use your own API key')
    const details = summary.closest('details')
    expect(details).not.toHaveAttribute('open')
  })

  it('stores the key in sessionStorage, not localStorage, and removes it on demand', async () => {
    const user = userEvent.setup()
    render(<App />)

    const summary = await screen.findByText('Use your own API key')
    await user.click(summary)

    const keyInput = await screen.findByPlaceholderText('sk-…')
    await user.type(keyInput, 'my-secret-key')

    expect(window.sessionStorage.getItem('smart-travel-byok')).toContain('my-secret-key')
    expect(window.localStorage.getItem('smart-travel-byok')).toBeNull()

    const removeButton = await screen.findByText('Remove key')
    await user.click(removeButton)

    expect(window.sessionStorage.getItem('smart-travel-byok')).toBeNull()
    expect(screen.queryByDisplayValue('my-secret-key')).not.toBeInTheDocument()
  })

  it('populates the provider/model select from GET /llm-options', async () => {
    render(<App />)

    const summary = await screen.findByText('Use your own API key')
    await screen.findByText(/gemini \/ gemini-3.1-flash-lite/)
    expect(summary).toBeInTheDocument()
    expect(await screen.findByText(/openai \/ gpt-5.4-nano/)).toBeInTheDocument()
  })
})
