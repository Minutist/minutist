import { useAppEventBridge } from "./shell/event-listener";
import { MainWindow } from "./shell/MainWindow";

/**
 * Application root.
 *
 * Mounts the global Tauri event bridge at the top of the tree so it is
 * never accidentally unmounted by conditional rendering in child components.
 */
export function App() {
  useAppEventBridge();
  return <MainWindow />;
}
