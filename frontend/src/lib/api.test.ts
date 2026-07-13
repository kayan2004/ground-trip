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
  it('sets the X-LLM-API-Key header when an apiKey is provided, and always sends credentials', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun(
      { prompt: 'a trip', retrieval_top_k: 3, llm_provider: 'openai', llm_model: 'gpt-5.4-nano' },
      'user-supplied-key',
    )

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBe('user-supplied-key')
    // Auth is a cookie now, not a header this client attaches - 'include'
    // is what makes the browser send it.
    expect(requestInit.credentials).toBe('include')
  })

  it('omits the X-LLM-API-Key header when no apiKey is provided', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun({ prompt: 'a trip', retrieval_top_k: 3 })

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBeUndefined()
  })
})
