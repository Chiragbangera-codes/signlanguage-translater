import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next blocks cross-origin requests to dev-only assets (including the HMR
  // socket) by default. Opening the app on a LAN address rather than localhost
  // trips that block, the dev client never connects, and the page paints
  // server HTML that never hydrates — visible as dead buttons. Listing the LAN
  // ranges keeps phone/second-machine testing working.
  //
  // Note: getUserMedia needs a secure context, and plain http:// on a LAN IP is
  // not one. Camera access works on localhost or over https only.
  allowedDevOrigins: ["192.168.*.*", "172.*.*.*", "10.*.*.*"],
};

export default nextConfig;
