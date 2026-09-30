# INTEL-I Vendor SDK adapters

This package is the controlled boundary for native vendor SDK integrations.

Each vendor adapter must:

1. subclass `VendorSDKAdapter`;
2. implement `resolve()`;
3. return `ConnectorResult`;
4. return either a normalized RTSP/HTTP/HLS source or a native frame source;
5. keep SDK credentials inside its in-memory configuration;
6. never log credentials or stream URLs.

Set `INTEL_I_SDK_MODULE_ALLOWLIST` to the exact adapter modules that are
approved for the deployment. The loader rejects modules outside this allowlist.

The generic framework is intentionally vendor-neutral because native SDK
packages and their licensing/runtime requirements differ by manufacturer.
