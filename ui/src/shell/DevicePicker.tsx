import { useRecordingStore } from "../state/recording";

/**
 * Labelled `<select>` for choosing the audio input device.
 *
 * The device list is populated from `state.devices`, which is populated by
 * `refreshDevices()` on mount (called from `MainWindow`). When the backend
 * emits a `devices_changed` event the store re-fetches automatically.
 */
export function DevicePicker() {
  const devices = useRecordingStore((s) => s.devices);
  const selectedDeviceId = useRecordingStore((s) => s.selectedDeviceId);
  const setSelectedDevice = useRecordingStore((s) => s.setSelectedDevice);

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const val = e.target.value;
    setSelectedDevice(val === "" ? null : val);
  }

  return (
    <div className="device-picker">
      <label htmlFor="device-select">Input device</label>
      <select
        id="device-select"
        value={selectedDeviceId ?? ""}
        onChange={handleChange}
      >
        <option value="">System default</option>
        {devices.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name}
            {d.is_default ? " (default)" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
