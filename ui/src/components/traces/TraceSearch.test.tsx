import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TraceSearch, { parseQuery } from './TraceSearch';

describe('parseQuery', () => {
  it('parses service filter', () => {
    const result = parseQuery('service:chat-api');
    expect(result.service).toBe('chat-api');
  });

  it('parses model filter', () => {
    const result = parseQuery('model:gpt-4');
    expect(result.model_id).toBe('gpt-4');
  });

  it('parses status filter', () => {
    const result = parseQuery('status:error');
    expect(result.status).toBe('error');
  });

  it('parses duration greater than filter', () => {
    const result = parseQuery('duration:>2000');
    expect(result.duration_min).toBe(2000);
  });

  it('parses duration less than filter', () => {
    const result = parseQuery('duration:<500');
    expect(result.duration_max).toBe(500);
  });

  it('parses time range filter', () => {
    const result = parseQuery('time:last-24h');
    expect(result.time_range).toBe('last-24h');
  });

  it('parses has:drift filter', () => {
    const result = parseQuery('has:drift');
    expect(result.has_drift).toBe(true);
  });

  it('parses has:safety filter', () => {
    const result = parseQuery('has:safety');
    expect(result.has_safety_issues).toBe(true);
  });

  it('parses user filter', () => {
    const result = parseQuery('user:user-123');
    expect(result.user_id).toBe('user-123');
  });

  it('captures free text', () => {
    const result = parseQuery('some free text');
    expect(result.text).toBe('some free text');
  });

  it('parses multiple filters', () => {
    const result = parseQuery('service:api model:gpt-4 status:ok duration:>1000');
    expect(result.service).toBe('api');
    expect(result.model_id).toBe('gpt-4');
    expect(result.status).toBe('ok');
    expect(result.duration_min).toBe(1000);
  });

  it('handles mixed filters and text', () => {
    const result = parseQuery('service:api hello world');
    expect(result.service).toBe('api');
    expect(result.text).toBe('hello world');
  });

  it('handles empty query', () => {
    const result = parseQuery('');
    expect(result).toEqual({});
  });
});

describe('TraceSearch', () => {
  it('renders search input', () => {
    render(
      <TraceSearch
        value=""
        onChange={() => {}}
        onSearch={() => {}}
        savedQueries={[]}
        recentQueries={[]}
      />
    );

    expect(screen.getByPlaceholderText(/Search traces/)).toBeInTheDocument();
  });

  it('calls onChange when typing', () => {
    const handleChange = vi.fn();

    render(
      <TraceSearch
        value=""
        onChange={handleChange}
        onSearch={() => {}}
        savedQueries={[]}
        recentQueries={[]}
      />
    );

    const input = screen.getByPlaceholderText(/Search traces/);
    fireEvent.change(input, { target: { value: 'test query' } });

    expect(handleChange).toHaveBeenCalledWith('test query');
  });

  it('calls onSearch when pressing Enter', () => {
    const handleSearch = vi.fn();

    render(
      <TraceSearch
        value="service:api"
        onChange={() => {}}
        onSearch={handleSearch}
        savedQueries={[]}
        recentQueries={[]}
      />
    );

    const input = screen.getByPlaceholderText(/Search traces/);
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(handleSearch).toHaveBeenCalled();
  });

  it('displays saved queries', () => {
    const savedQueries = [
      { name: 'Errors Today', query: 'status:error time:last-24h' },
      { name: 'Slow Requests', query: 'duration:>2000' },
    ];

    render(
      <TraceSearch
        value=""
        onChange={() => {}}
        onSearch={() => {}}
        savedQueries={savedQueries}
        recentQueries={[]}
      />
    );

    // Focus the input to show dropdown
    const input = screen.getByPlaceholderText(/Search traces/);
    fireEvent.focus(input);

    expect(screen.getByText('Errors Today')).toBeInTheDocument();
    expect(screen.getByText('Slow Requests')).toBeInTheDocument();
  });
});
