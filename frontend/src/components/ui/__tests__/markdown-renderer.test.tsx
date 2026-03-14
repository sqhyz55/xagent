/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MarkdownRenderer } from '../markdown-renderer'

describe('MarkdownRenderer', () => {
  it('renders inline math with KaTeX without leaving dollar delimiters', () => {
    const content = 'The equation is $x^2 + y^2 = 1$.'
    render(<MarkdownRenderer content={content} />)

    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBeGreaterThan(0)
    expect(screen.queryByText(/\$x\^2 \+ y\^2 = 1\$/)).toBeNull()
  })

  it('does not treat $PATH inside code block as math', () => {
    const content = '```bash\necho $PATH\n```'
    render(<MarkdownRenderer content={content} />)

    const pre = screen.getByText(/echo \$PATH/)
    expect(pre).toBeInTheDocument()
    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBe(0)
  })

  it('does not treat $HOME inside inline code as math', () => {
    const content = 'Use `echo $HOME` to see your home dir.'
    render(<MarkdownRenderer content={content} />)

    const code = screen.getByText('echo $HOME')
    expect(code.tagName.toLowerCase()).toBe('code')
    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBe(0)
  })

  it('handles file: links with onFileClick callback', () => {
    const handleFileClick = vi.fn()
    const content = '[open file](file:/tmp/test.txt)'

    render(<MarkdownRenderer content={content} onFileClick={handleFileClick} />)

    const link = screen.getByText('open file')
    fireEvent.click(link)

    expect(handleFileClick).toHaveBeenCalledTimes(1)
    expect(handleFileClick).toHaveBeenCalledWith('/tmp/test.txt', 'open file')
  })
})
