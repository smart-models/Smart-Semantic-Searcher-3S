const rawSeconds = process.env.QHM_E2E_COOLDOWN_SECONDS ?? "15";
const seconds = Number(rawSeconds);

if (!Number.isFinite(seconds) || seconds < 0) {
  console.error("QHM_E2E_COOLDOWN_SECONDS must be a non-negative number.");
  process.exit(2);
}

if (seconds > 0) {
  console.log(
    `Waiting ${seconds}s for the middleware token bucket to recover after setup...`,
  );
  await new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}
