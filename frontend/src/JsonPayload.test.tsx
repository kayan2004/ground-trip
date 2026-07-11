import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { JsonPayload } from './JsonPayload'

describe('JsonPayload', () => {
  it('pretty-prints valid JSON with indentation', () => {
    const { container } = render(
      <JsonPayload value={JSON.stringify({ destination: 'Bariloche', score: 0.87 })} />,
    )

    const pre = container.querySelector('pre.json-payload')
    expect(pre).not.toBeNull()
    expect(pre?.textContent).toContain('"destination": "Bariloche"')
    expect(pre?.textContent).toContain('"score": 0.87')
  })

  it('color-codes keys, strings, numbers, booleans, and null distinctly', () => {
    const { container } = render(
      <JsonPayload
        value={JSON.stringify({
          name: 'Zermatt',
          rank: 2,
          region_match: true,
          budget_delta: null,
        })}
      />,
    )

    expect(container.querySelector('.json-payload-key')?.textContent).toContain('"name"')
    expect(container.querySelector('.json-payload-string')?.textContent).toContain('Zermatt')
    expect(container.querySelector('.json-payload-number')?.textContent).toBe('2')
    expect(container.querySelector('.json-payload-boolean')?.textContent).toBe('true')
    expect(container.querySelector('.json-payload-null')?.textContent).toBe('null')
  })

  it('falls back to plain text for non-JSON payloads', () => {
    render(<JsonPayload value="Destination recommendation failed: OSError: boom" />)

    expect(
      screen.getByText('Destination recommendation failed: OSError: boom'),
    ).toBeInTheDocument()
  })

  it('shows an empty placeholder for a blank payload', () => {
    render(<JsonPayload value="" />)

    expect(screen.getByText('(empty)')).toBeInTheDocument()
  })
})
