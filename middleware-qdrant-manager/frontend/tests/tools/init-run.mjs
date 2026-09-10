// Register absent exact fixture names once, before any campaign mutation.
import { fixtureName, initializeManifest, setupCall, manifestPath } from '../harness.mjs';
import { FIXTURES } from '../fixture-catalog.mjs';

const { data } = await setupCall('GET', '/collections');
if (!Array.isArray(data.collections) || data.collections.some((item) => typeof item !== 'string')) {
  throw new Error('Malformed collection inventory; no campaign registered.');
}
const manifest = initializeManifest(data.collections, FIXTURES.map(fixtureName));
console.log(`Registered ${manifest.ownedCollections.length} absent targets; protected ${manifest.initialCollections.length} existing collections. Manifest: ${manifestPath}`);
