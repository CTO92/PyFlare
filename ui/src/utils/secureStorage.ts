/**
 * Secure Storage Utility
 * Provides encrypted storage for sensitive data when httpOnly cookies are not available
 *
 * Security measures:
 * - Uses Web Crypto API for encryption
 * - Derives encryption key from a combination of factors
 * - Adds integrity verification
 * - Implements automatic expiration
 */

const ENCRYPTION_ALGORITHM = 'AES-GCM';
const KEY_DERIVATION_ALGORITHM = 'PBKDF2';
const KEY_LENGTH = 256;
const ITERATIONS = 100000;

interface EncryptedData {
  iv: string;
  data: string;
  tag: string;
  salt: string;
  expiresAt?: number;
}

/**
 * Generate a device fingerprint for key derivation
 * This adds an extra layer of security - stolen encrypted data
 * won't decrypt on a different device
 */
function getDeviceFingerprint(): string {
  const components = [
    navigator.userAgent,
    navigator.language,
    screen.colorDepth.toString(),
    screen.width.toString(),
    screen.height.toString(),
    new Date().getTimezoneOffset().toString(),
  ];
  return components.join('|');
}

/**
 * Derive an encryption key from the device fingerprint
 */
async function deriveKey(salt: Uint8Array): Promise<CryptoKey> {
  const fingerprint = getDeviceFingerprint();
  const encoder = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(fingerprint),
    KEY_DERIVATION_ALGORITHM,
    false,
    ['deriveBits', 'deriveKey']
  );

  return crypto.subtle.deriveKey(
    {
      name: KEY_DERIVATION_ALGORITHM,
      salt,
      iterations: ITERATIONS,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: ENCRYPTION_ALGORITHM, length: KEY_LENGTH },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Convert ArrayBuffer to base64 string
 */
function bufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert base64 string to ArrayBuffer
 */
function base64ToBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Encrypt sensitive data
 */
async function encrypt(data: string, expiresAt?: number): Promise<string> {
  const encoder = new TextEncoder();
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(salt);

  const encrypted = await crypto.subtle.encrypt(
    { name: ENCRYPTION_ALGORITHM, iv },
    key,
    encoder.encode(data)
  );

  const encryptedData: EncryptedData = {
    iv: bufferToBase64(iv.buffer),
    data: bufferToBase64(encrypted),
    tag: '', // GCM includes tag in ciphertext
    salt: bufferToBase64(salt.buffer),
    expiresAt,
  };

  return btoa(JSON.stringify(encryptedData));
}

/**
 * Decrypt sensitive data
 */
async function decrypt(encryptedString: string): Promise<string | null> {
  try {
    const encryptedData: EncryptedData = JSON.parse(atob(encryptedString));

    // Check expiration
    if (encryptedData.expiresAt && Date.now() > encryptedData.expiresAt) {
      return null;
    }

    const salt = new Uint8Array(base64ToBuffer(encryptedData.salt));
    const iv = new Uint8Array(base64ToBuffer(encryptedData.iv));
    const data = base64ToBuffer(encryptedData.data);
    const key = await deriveKey(salt);

    const decrypted = await crypto.subtle.decrypt(
      { name: ENCRYPTION_ALGORITHM, iv },
      key,
      data
    );

    const decoder = new TextDecoder();
    return decoder.decode(decrypted);
  } catch {
    return null;
  }
}

/**
 * Secure storage class for managing encrypted data in sessionStorage
 * Using sessionStorage instead of localStorage for additional security:
 * - Data is cleared when browser/tab is closed
 * - Not shared across tabs (reduces attack surface)
 */
export class SecureStorage {
  private prefix: string;

  constructor(prefix: string = 'pyflare_secure_') {
    this.prefix = prefix;
  }

  /**
   * Store data securely
   * @param key Storage key
   * @param value Value to store
   * @param ttlMs Time to live in milliseconds (optional)
   */
  async setItem(key: string, value: string, ttlMs?: number): Promise<void> {
    const expiresAt = ttlMs ? Date.now() + ttlMs : undefined;
    const encrypted = await encrypt(value, expiresAt);
    sessionStorage.setItem(this.prefix + key, encrypted);
  }

  /**
   * Retrieve data securely
   * @param key Storage key
   * @returns Decrypted value or null if not found/expired/invalid
   */
  async getItem(key: string): Promise<string | null> {
    const encrypted = sessionStorage.getItem(this.prefix + key);
    if (!encrypted) {
      return null;
    }

    const decrypted = await decrypt(encrypted);
    if (decrypted === null) {
      // Data is invalid or expired, remove it
      this.removeItem(key);
    }
    return decrypted;
  }

  /**
   * Remove item from storage
   */
  removeItem(key: string): void {
    sessionStorage.removeItem(this.prefix + key);
  }

  /**
   * Clear all items with this prefix
   */
  clear(): void {
    const keysToRemove: string[] = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key?.startsWith(this.prefix)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => sessionStorage.removeItem(key));
  }
}

/**
 * Check if Web Crypto API is available
 */
export function isSecureStorageAvailable(): boolean {
  return !!(
    typeof crypto !== 'undefined' &&
    crypto.subtle &&
    typeof crypto.subtle.encrypt === 'function'
  );
}

/**
 * Singleton instance for the application
 */
export const secureStorage = new SecureStorage();
