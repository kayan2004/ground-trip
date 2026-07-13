import { afterEach, describe, expect, it, vi } from 'vitest'

import { createAgentRun } from './api'

function mockFetchOnce(body: unknown, ok = true) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status: ok ? 201 : 400,
    json: () => Promise.resolve(body),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('createAgentRun', () => {
  it('sets the X-LLM-API-Key header when an apiKey is provided', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun(
      'jwt-token',
      { prompt: 'a trip', retrieval_top_k: 3, llm_provider: 'openai', llm_model: 'gpt-5.4-nano' },
      'user-supplied-key',
    )

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBe('user-supplied-key')
    expect(requestInit.headers['Authorization']).toBe('Bearer jwt-token')
  })

  it('omits the X-LLM-API-Key header when no apiKey is provided', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun('jwt-token', { prompt: 'a trip', retrieval_top_k: 3 })

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBeUndefined()
  })
})
