/**
 * CSRF Protection Utility
 * Implements double-submit cookie pattern for CSRF protection
 */

const CSRF_COOKIE_NAME = 'pyflare-csrf-token';
const CSRF_HEADER_NAME = 'X-CSRF-Token';

/**
 * Generate a cryptographically secure random token
 */
function generateToken(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Get or create CSRF token
 * Uses double-submit cookie pattern:
 * 1. Token is stored in a cookie (SameSite=Strict)
 * 2. Token is also sent in request header
 * 3. Server verifies both match
 */
export function getCSRFToken(): string {
  // Try to get existing token from cookie
  const cookies = document.cookie.split(';');
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === CSRF_COOKIE_NAME && value) {
      return value;
    }
  }

  // Generate new token if not found
  const token = generateToken();

  // Set cookie with secure flags
  // SameSite=Strict prevents CSRF attacks
  // Secure flag ensures HTTPS only in production
  const isSecure = window.location.protocol === 'https:';
  const secureFlag = isSecure ? '; Secure' : '';
  document.cookie = `${CSRF_COOKIE_NAME}=${token}; SameSite=Strict; Path=/${secureFlag}`;

  return token;
}

/**
 * Get CSRF header name
 */
export function getCSRFHeaderName(): string {
  return CSRF_HEADER_NAME;
}

/**
 * Create headers object with CSRF token included
 */
export function getCSRFHeaders(): Record<string, string> {
  return {
    [CSRF_HEADER_NAME]: getCSRFToken(),
  };
}

/**
 * Clear CSRF token (e.g., on logout)
 */
export function clearCSRFToken(): void {
  document.cookie = `${CSRF_COOKIE_NAME}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/`;
}

/**
 * Refresh CSRF token (e.g., after sensitive operations)
 */
export function refreshCSRFToken(): string {
  clearCSRFToken();
  return getCSRFToken();
}
