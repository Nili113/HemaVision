export const GUEST_ANALYSIS_LIMIT = 3;
export const GUEST_ANALYSIS_COUNT_KEY = 'hemavision_guest_analysis_count';

export interface GuestUsage {
  used: number;
  remaining: number;
  limit: number;
}

export function getGuestUsedCount(): number {
  if (typeof window === 'undefined') return 0;
  const raw = localStorage.getItem(GUEST_ANALYSIS_COUNT_KEY);
  const parsed = raw ? Number.parseInt(raw, 10) : 0;
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.min(parsed, GUEST_ANALYSIS_LIMIT);
}

export function getGuestUsage(): GuestUsage {
  const used = getGuestUsedCount();
  return {
    used,
    remaining: Math.max(0, GUEST_ANALYSIS_LIMIT - used),
    limit: GUEST_ANALYSIS_LIMIT,
  };
}

export function incrementGuestUsage(): GuestUsage {
  if (typeof window === 'undefined') {
    return { used: 0, remaining: GUEST_ANALYSIS_LIMIT, limit: GUEST_ANALYSIS_LIMIT };
  }

  const current = getGuestUsedCount();
  const next = Math.min(current + 1, GUEST_ANALYSIS_LIMIT);
  localStorage.setItem(GUEST_ANALYSIS_COUNT_KEY, String(next));
  return {
    used: next,
    remaining: Math.max(0, GUEST_ANALYSIS_LIMIT - next),
    limit: GUEST_ANALYSIS_LIMIT,
  };
}

export function resetGuestUsage(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(GUEST_ANALYSIS_COUNT_KEY);
}
