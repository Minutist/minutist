/** The message of a caught `unknown` value, for surfacing in store `lastError` fields. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
