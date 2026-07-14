import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

const sampleUser = {
  id: 1,
  email: 'traveler@test.com',
  full_name: 'Traveler',
  is_active: true,
  created_at: '2026-01-01T00:00:00Z',
}

const sampleHistory = [
  { id: 42, prompt: 'A relaxing week in the mountains', status: 'completed', created_at: '2026-07-14T10:00:00Z' },
  { id: 41, prompt: 'A budget beach trip', status: 'partial', created_at: '2026-07-13T10:00:00Z' },
]

const sampleDetail = {
  id: 41,
  user_id: 1,
  prompt: 'A budget beach trip',
  response: 'Try Bariloche in the off season.',
  status: 'partial',
  created_at: '2026-07-13T10:00:00Z',
  tool_logs: [],
  recommendations: [],
}

function mockFetch(options: { historyEmpty?: boolean } = {}) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url.includes('/llm-options')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
    }
    if (url.includes('/auth/me')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(sampleUser) })
    }
    if (/\/agent-runs\/\d+$/.test(url)) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(sampleDetail) })
    }
    if (url.includes('/agent-runs')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(options.historyEmpty ? [] : sampleHistory),
      })
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  window.sessionStorage.clear()
  window.history.pushState(null, '', '/app')
})

afterEach(() => {
  vi.unstubAllGlobals()
  window.sessionStorage.clear()
})

describe('History panel', () => {
  it('lists past trip plans fetched from GET /agent-runs', async () => {
    mockFetch()
    render(<App />)

    expect(await screen.findByText('A relaxing week in the mountains')).toBeInTheDocument()
    expect(await screen.findByText('A budget beach trip')).toBeInTheDocument()
    expect(screen.getByText('2 saved')).toBeInTheDocument()
  })

  it('shows an empty state when there is no history yet', async () => {
    mockFetch({ historyEmpty: true })
    render(<App />)

    expect(
      await screen.findByText('No trips yet - your plans will show up here.'),
    ).toBeInTheDocument()
  })

  it('loads full detail via GET /agent-runs/{id} when a history row is clicked', async () => {
    const fetchMock = mockFetch()
    const user = userEvent.setup()
    render(<App />)

    const row = await screen.findByText('A budget beach trip')
    await user.click(row)

    expect(await screen.findByText('Try Bariloche in the off season.')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/agent-runs\/41$/),
      expect.anything(),
    )
  })
})
