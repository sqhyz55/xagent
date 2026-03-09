import React from 'react'
import { marked } from 'marked'
import { getApiUrl } from '@/lib/utils'

// Enhanced Markdown detection function: covers broader Markdown features not limited to starting with #
const isLikelyMarkdown = (s: string): boolean => {
  const t = s.trim()
  if (!t) return false
  return (
    t.startsWith('#') || // Heading
    s.includes('```') || // Code block
    /(\n|^)\s*(-|\*|\d+\.)\s/.test(s) || // List (unordered/ordered)
    (s.includes('|') && s.includes('---')) || // Table
    /\[[^\]]+\]\([^\)]+\)/.test(s) || // Link [text](url)
    /!\[[^\]]*\]\([^\)]+\)/.test(s) || // Image ![alt](url)
    /(\n|^)\s*>\s/.test(s) || // Blockquote
    /(\n|^)\s*---\s*(\n|$)/.test(s) // Horizontal rule
  )
}

interface MarkdownRendererProps {
  content: string
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
}

export function MarkdownRenderer({ content, className = '', onFileClick }: MarkdownRendererProps) {
  const [html, setHtml] = React.useState('')
  const containerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const parseMarkdown = async () => {
      try {
        // Custom renderer for file: protocol links and images
        const renderer = new marked.Renderer()
        const defaultLinkRenderer = renderer.link.bind(renderer)
        const defaultImageRenderer = renderer.image.bind(renderer)

        renderer.link = (href: string | null, title: string | null, text: string) => {
          // Check if this is a file: protocol link
          if (href && href.startsWith('file:')) {
            const filePath = href.replace(/^file:/, '')
            // Return a data-link attribute so we can handle it with event delegation
            return `<a href="#" data-file-path="${filePath}" class="file-link" title="${title || ''}">${text}</a>`
          }
          // Use default renderer for other links
          return defaultLinkRenderer(href, title, text)
        }

        renderer.image = (href: string | null, title: string | null, text: string) => {
          // Handle image links with file: protocol
          if (href && href.startsWith('file:')) {
            const filePath = href.replace(/^file:/, '')
            const apiUrl = getApiUrl()

            // Extract task_id from filePath (e.g., "web_task_158/output/file.png" -> "158")
            const taskIdMatch = filePath.match(/web_task_(\d+)/)
            const taskId = taskIdMatch ? taskIdMatch[1] : null

            let imageUrl: string
            if (taskId) {
              // Use public preview API (no auth required)
              imageUrl = `${apiUrl}/api/files/public/preview/${taskId}/${encodeURIComponent(filePath)}`
            } else {
              // Fallback to download API (requires auth, won't work in img tag)
              imageUrl = `${apiUrl}/api/files/download/${encodeURIComponent(filePath)}`
            }

            // Also add data-file-path for click preview
            return `<img src="${imageUrl}" alt="${text || ''}" title="${title || text || ''}" data-file-path="${filePath}" class="file-image cursor-pointer" />`
          }
          // Use default renderer for other images
          return defaultImageRenderer(href, title, text)
        }

        // Use marked.use() to configure renderer (marked 5.x API)
        marked.use({ renderer })
        const parsed = await marked.parse(content)
        setHtml(parsed)
      } catch (error) {
        console.error('Error parsing markdown:', error)
        setHtml(content)
      }
    }

    parseMarkdown()
  }, [content])

  // Handle file link clicks and image clicks
  React.useEffect(() => {
    const container = containerRef.current
    if (!container || !onFileClick) return

    const handleFileClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement

      // Handle file link clicks
      const link = target.closest('.file-link') as HTMLAnchorElement
      if (link) {
        e.preventDefault()
        const filePath = link.getAttribute('data-file-path')
        if (filePath) {
          // Extract filename from path
          const fileName = filePath.split('/').pop() || filePath
          onFileClick(filePath, fileName)
        }
        return
      }

      // Handle image clicks with data-file-path attribute
      const img = target as HTMLImageElement
      if (img.tagName === 'IMG' && img.hasAttribute('data-file-path')) {
        e.preventDefault()
        const filePath = img.getAttribute('data-file-path')
        if (filePath) {
          // Extract filename from path
          const fileName = filePath.split('/').pop() || filePath
          onFileClick(filePath, fileName)
        }
      }
    }

    container.addEventListener('click', handleFileClick)
    return () => {
      container.removeEventListener('click', handleFileClick)
    }
  }, [onFileClick])

  return (
    <div
      ref={containerRef}
      className={`prose prose-invert max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
      style={{
        // Style file links differently
        '--link-color': '#3b82f6'
      } as React.CSSProperties}
    />
  )
}

interface JsonRendererProps {
  data: any
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
}

export function JsonRenderer({ data, className = '', onFileClick }: JsonRendererProps) {
  const [expanded, setExpanded] = React.useState(true)

  if (typeof data === 'string') {
    // Try to parse as JSON first
    try {
      const parsed = JSON.parse(data)
      return <JsonRenderer data={parsed} className={className} onFileClick={onFileClick} />
    } catch {
      // Render non-JSON string as markdown by default, so headings/lists/code blocks
      // from agent output are displayed as formatted content instead of raw source.
      return <MarkdownRenderer content={data} className={className} onFileClick={onFileClick} />
    }
  }

  if (typeof data === 'object' && data !== null) {
    // Check if it's a result object with output that might be markdown
    if (data.output && typeof data.output === 'string' && isLikelyMarkdown(data.output.trim())) {
      return (
        <div className={`space-y-3 ${className}`}>
          <div className="bg-muted p-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap">
            <div className="text-green-400 mb-2">✅ Task completed successfully</div>
            <div className="text-gray-400">Goal: {data.goal}</div>
          </div>
          <div className="border-t border-border pt-3">
            <div className="text-sm font-medium text-foreground mb-2">Result:</div>
            <MarkdownRenderer content={data.output} onFileClick={onFileClick} />
          </div>
        </div>
      )
    }

    // For other objects, display as formatted JSON
    return (
      <div className={`space-y-2 ${className}`}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          {expanded ? '▼' : '▶'} JSON Data
        </button>
        {expanded && (
          <pre className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  // For other types, display as string
  return (
    <pre className={`bg-muted py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
      {String(data)}
    </pre>
  )
}
