interface JsonPayloadProps {
  value: string
}

type JsonTokenType = 'key' | 'string' | 'number' | 'boolean' | 'null' | 'punctuation'

interface JsonToken {
  text: string
  type: JsonTokenType
}

// Matches one JSON value/key at a time: quoted strings (optionally followed
// by a colon, which marks them as an object key), true/false, null, or a
// number. Everything between matches (braces, brackets, commas, colons,
// whitespace/indentation) is left as plain punctuation.
const JSON_TOKEN_PATTERN =
  /"(?:\\u[a-fA-F0-9]{4}|\\.|[^"\\])*"(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g

function tokenizeJson(pretty: string): JsonToken[] {
  const tokens: JsonToken[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  JSON_TOKEN_PATTERN.lastIndex = 0
  while ((match = JSON_TOKEN_PATTERN.exec(pretty)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: pretty.slice(lastIndex, match.index), type: 'punctuation' })
    }

    const text = match[0]
    let type: JsonTokenType
    if (text.startsWith('"')) {
      type = /:\s*$/.test(text) ? 'key' : 'string'
    } else if (text === 'true' || text === 'false') {
      type = 'boolean'
    } else if (text === 'null') {
      type = 'null'
    } else {
      type = 'number'
    }

    tokens.push({ text, type })
    lastIndex = JSON_TOKEN_PATTERN.lastIndex
  }

  if (lastIndex < pretty.length) {
    tokens.push({ text: pretty.slice(lastIndex), type: 'punctuation' })
  }

  return tokens
}

/** Renders a tool-log payload string. Pretty-prints and syntax-colors it if
 * it's valid JSON (the common case - extraction/recommendation/RAG payloads
 * all are); falls back to plain text for payloads that aren't JSON (a raw
 * prompt, a plain-English error message, an empty string). */
export function JsonPayload({ value }: JsonPayloadProps) {
  const trimmed = value.trim()

  if (!trimmed) {
    return <p className="json-payload-empty">(empty)</p>
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    return <pre className="json-payload-raw">{value}</pre>
  }

  const pretty = JSON.stringify(parsed, null, 2)
  const tokens = tokenizeJson(pretty)

  return (
    <pre className="json-payload">
      {tokens.map((token, index) =>
        token.type === 'punctuation' ? (
          <span key={index}>{token.text}</span>
        ) : (
          <span key={index} className={`json-payload-${token.type}`}>
            {token.text}
          </span>
        ),
      )}
    </pre>
  )
}
