import { useState } from "react";

import UploadVideo from "../components/UploadVideo";
import CameraOnboarding from "../components/CameraOnboarding";


const CameraSetup = () => {
  const [
    activeTab,
    setActiveTab,
  ] = useState("onboarding");

  const handleTabChange = (
    tab
  ) => {
    if (
      tab !== "onboarding" &&
      tab !== "upload"
    ) {
      return;
    }

    setActiveTab(tab);
  };


  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="w-full bg-[#0b1b2e] border border-[#1f4b7a] rounded-lg p-4 sm:p-5 text-white">

      {/* ====================================================== */}
      {/* TAB NAVIGATION */}
      {/* ====================================================== */}

      <div
        className="grid grid-cols-2 border-b border-[#1f4b7a] mb-5"
        role="tablist"
        aria-label="Camera setup modes"
      >

        {/* ==================================================== */}
        {/* LIVE RTSP TAB */}
        {/* ==================================================== */}

        <button
          type="button"
          role="tab"
          aria-selected={
            activeTab === "onboarding"
          }
          aria-controls="live-camera-panel"
          onClick={() =>
            handleTabChange(
              "onboarding"
            )
          }
          className={`
            py-3
            text-lg
            font-bold
            transition
            duration-200
            rounded-t-md
            focus:outline-none
            focus:ring-2
            focus:ring-blue-500
            ${
              activeTab === "onboarding"
                ? "bg-blue-600 text-white"
                : "text-[#49759c] hover:text-white hover:bg-[#102943]"
            }
          `}
        >
          Camera Onboarding & Fleet
        </button>


        {/* ==================================================== */}
        {/* UPLOAD VIDEO TAB */}
        {/* ==================================================== */}

        <button
          type="button"
          role="tab"
          aria-selected={
            activeTab === "upload"
          }
          aria-controls="upload-video-panel"
          onClick={() =>
            handleTabChange(
              "upload"
            )
          }
          className={`
            py-3
            text-lg
            font-bold
            transition
            duration-200
            rounded-t-md
            focus:outline-none
            focus:ring-2
            focus:ring-blue-500
            ${
              activeTab === "upload"
                ? "bg-blue-600 text-white"
                : "text-[#49759c] hover:text-white hover:bg-[#102943]"
            }
          `}
        >
          Upload Video
        </button>

      </div>


      {/* ====================================================== */}
      {/* LIVE CAMERA SETUP */}
      {/* ====================================================== */}

      {activeTab === "onboarding" && (
        <div
          id="camera-onboarding-panel"
          role="tabpanel"
          aria-labelledby="live-camera-tab"
        >
          <CameraOnboarding />
        </div>
      )}


      {/* ====================================================== */}
      {/* UPLOAD VIDEO SETUP */}
      {/* ====================================================== */}

      {activeTab === "upload" && (
        <div
          id="upload-video-panel"
          role="tabpanel"
          aria-labelledby="upload-video-tab"
        >
          <UploadVideo />
        </div>
      )}

    </div>
  );
};


export default CameraSetup;
