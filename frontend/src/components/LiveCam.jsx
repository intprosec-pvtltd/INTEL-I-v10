import React, {
  useCallback,
  useState,
} from "react";

import api from "../api/axios";
import toast from "react-hot-toast";

import GISMap from "./GisMap";

// ============================================================
// CONSTANTS
// ============================================================

const MIN_LATITUDE = -90;
const MAX_LATITUDE = 90;

const MIN_LONGITUDE = -180;
const MAX_LONGITUDE = 180;

const MAX_CAMERA_NAME_LENGTH = 150;
const MAX_LOCATION_NAME_LENGTH = 255;
const MAX_SOURCE_LENGTH = 2000;

// ============================================================
// SOURCE TYPES
// ============================================================

const SOURCE_TYPES = {
  RTSP: "rtsp",
  HTTP: "http",
  HTTPS: "https",
  HLS: "hls",
  WEBCAM: "webcam",
};

// ============================================================
// VALIDATION
// ============================================================

const isValidLatitude = (value) => {
  const number = Number(value);

  return (
    Number.isFinite(number) &&
    number >= MIN_LATITUDE &&
    number <= MAX_LATITUDE
  );
};

const isValidLongitude = (value) => {
  const number = Number(value);

  return (
    Number.isFinite(number) &&
    number >= MIN_LONGITUDE &&
    number <= MAX_LONGITUDE
  );
};

const validateSourceUrl = (value, sourceType) => {
  const cleanValue = value.trim();

  if (!cleanValue) {
    return false;
  }

  if (cleanValue.length > MAX_SOURCE_LENGTH) {
    return false;
  }

  // Webcam uses a device index such as "0".
  if (sourceType === SOURCE_TYPES.WEBCAM) {
    return /^\d+$/.test(cleanValue);
  }

  try {
    const url = new URL(cleanValue);

    if (
      sourceType === SOURCE_TYPES.RTSP &&
      url.protocol !== "rtsp:"
    ) {
      return false;
    }

    if (
      sourceType === SOURCE_TYPES.HTTP &&
      url.protocol !== "http:"
    ) {
      return false;
    }

    if (
      sourceType === SOURCE_TYPES.HTTPS &&
      url.protocol !== "https:"
    ) {
      return false;
    }

    if (sourceType === SOURCE_TYPES.HLS) {
      if (
        url.protocol !== "http:" &&
        url.protocol !== "https:"
      ) {
        return false;
      }

      if (
        !url.pathname
          .toLowerCase()
          .endsWith(".m3u8")
      ) {
        return false;
      }
    }

    return true;
  } catch {
    return false;
  }
};

// ============================================================
// LIVE CAMERA SETUP
// ============================================================

const LiveCam = () => {
  // ==========================================================
  // REQUIRED CAMERA DATA
  // ==========================================================

  const [cameraName, setCameraName] = useState("");

  const [sourceType, setSourceType] = useState(
    SOURCE_TYPES.RTSP
  );

  const [source, setSource] = useState("");

  // ==========================================================
  // CAMERA LOCATION
  // ==========================================================

  const [locationName, setLocationName] = useState("");

  // ==========================================================
  // GIS LOCATION
  // ==========================================================

  const [latitude, setLatitude] = useState("");

  const [longitude, setLongitude] = useState("");

  const [selectedLocation, setSelectedLocation] =
    useState(null);

  // ==========================================================
  // UI STATE
  // ==========================================================

  const [loading, setLoading] = useState(false);

  // ==========================================================
  // MAP CLICK
  // ==========================================================

  const handleMapClick = useCallback((location) => {
    if (!location) {
      return;
    }

    const lat = Number(location.latitude);
    const lng = Number(location.longitude);

    if (
      !isValidLatitude(lat) ||
      !isValidLongitude(lng)
    ) {
      toast.error("Invalid map coordinates");
      return;
    }

    const normalizedLat = Number(lat.toFixed(6));
    const normalizedLng = Number(lng.toFixed(6));

    setLatitude(String(normalizedLat));
    setLongitude(String(normalizedLng));

    setSelectedLocation({
      latitude: normalizedLat,
      longitude: normalizedLng,
    });

    toast.success(
      `Location selected: ${normalizedLat}, ${normalizedLng}`
    );
  }, []);

  // ==========================================================
  // SOURCE TYPE CHANGE
  // ==========================================================

  const handleSourceTypeChange = useCallback((event) => {
    const nextType = event.target.value;

    if (
      !Object.values(SOURCE_TYPES).includes(nextType)
    ) {
      return;
    }

    setSourceType(nextType);
    setSource("");
  }, []);

  // ==========================================================
  // SOURCE PLACEHOLDER
  // ==========================================================

  const getSourcePlaceholder = useCallback(() => {
    switch (sourceType) {
      case SOURCE_TYPES.HTTP:
        return "http://192.168.1.100:8080/video";

      case SOURCE_TYPES.HTTPS:
        return "https://example.com/camera/video";

      case SOURCE_TYPES.HLS:
        return "https://example.com/live/stream/index.m3u8";

      case SOURCE_TYPES.WEBCAM:
        return "0";

      case SOURCE_TYPES.RTSP:
      default:
        return "rtsp://camera-host:554/stream";
    }
  }, [sourceType]);

  // ==========================================================
  // SAVE CAMERA
  // ==========================================================

  const saveCamera = async () => {
    if (loading) {
      return;
    }

    // ========================================================
    // CAMERA NAME
    // ========================================================

    const cleanCameraName = cameraName.trim();

    if (!cleanCameraName) {
      toast.error("Enter camera name");
      return;
    }

    if (
      cleanCameraName.length >
      MAX_CAMERA_NAME_LENGTH
    ) {
      toast.error("Camera name is too long");
      return;
    }

    // ========================================================
    // CAMERA LOCATION
    // ========================================================

    const cleanLocationName =
      locationName.trim();

    if (!cleanLocationName) {
      toast.error("Enter camera location");
      return;
    }

    if (
      cleanLocationName.length >
      MAX_LOCATION_NAME_LENGTH
    ) {
      toast.error("Camera location is too long");
      return;
    }

    // ========================================================
    // CAMERA SOURCE
    // ========================================================

    const cleanSource = source.trim();

    if (!cleanSource) {
      toast.error("Enter the camera source");
      return;
    }

    if (
      !validateSourceUrl(
        cleanSource,
        sourceType
      )
    ) {
      if (sourceType === SOURCE_TYPES.HLS) {
        toast.error(
          "Enter a valid HLS .m3u8 URL"
        );
      } else if (
        sourceType === SOURCE_TYPES.WEBCAM
      ) {
        toast.error(
          "Enter a valid webcam device number"
        );
      } else {
        toast.error(
          "Enter a valid camera stream URL"
        );
      }

      return;
    }

    // ========================================================
    // GIS LATITUDE
    // ========================================================

    if (latitude.trim() === "") {
      toast.error(
        "Select the camera location on the GIS map"
      );
      return;
    }

    // ========================================================
    // GIS LONGITUDE
    // ========================================================

    if (longitude.trim() === "") {
      toast.error(
        "Select the camera location on the GIS map"
      );
      return;
    }

    const lat = Number(latitude);
    const lng = Number(longitude);

    if (!isValidLatitude(lat)) {
      toast.error(
        "Latitude must be between -90 and 90"
      );
      return;
    }

    if (!isValidLongitude(lng)) {
      toast.error(
        "Longitude must be between -180 and 180"
      );
      return;
    }

    // ========================================================
    // ENSURE GIS MAP WAS USED
    // ========================================================

    if (
      !selectedLocation ||
      Number(selectedLocation.latitude) !== Number(lat) ||
      Number(selectedLocation.longitude) !== Number(lng)
    ) {
      toast.error(
        "Select the camera location on the GIS map"
      );
      return;
    }

    // ========================================================
    // NORMALIZE GIS COORDINATES
    // ========================================================

    const normalizedLatitude =
      Number(lat.toFixed(6));

    const normalizedLongitude =
      Number(lng.toFixed(6));

    // ========================================================
    // CORE CAMERA PAYLOAD
    //
    // Only the fields required by the simplified
    // operator-facing camera setup are sent.
    //
    // Technical metadata such as FPS, resolution,
    // codec, transport, heading and FOV are intentionally
    // not collected here.
    // ========================================================

    const payload = {
      camera_name: cleanCameraName,
      source: cleanSource,
      source_type: sourceType,
      location_name: cleanLocationName,
      latitude: normalizedLatitude,
      longitude: normalizedLongitude,
    };

    // ========================================================
    // SAVE TO BACKEND
    // ========================================================

    try {
      setLoading(true);

      const response = await api.post(
        "/cameras",
        payload
      );

      const savedCamera = response?.data;

      // Do not log source URLs or credentials.
      console.debug(
        "Camera created successfully",
        {
          cam_id:
            savedCamera?.cam_id ||
            savedCamera?.camera_id,

          latitude:
            savedCamera?.latitude,

          longitude:
            savedCamera?.longitude,

          location_name:
            savedCamera?.location_name,
        }
      );

      toast.success(
        "Camera saved successfully"
      );

      // ======================================================
      // RESET FORM
      // ======================================================

      setCameraName("");

      setSourceType(
        SOURCE_TYPES.RTSP
      );

      setSource("");

      setLocationName("");

      setLatitude("");

      setLongitude("");

      setSelectedLocation(null);
    } catch (error) {
      console.error(
        "Failed to save camera",
        {
          status:
            error?.response?.status,
        }
      );

      const status =
        error?.response?.status;

      if (
        status === 400 ||
        status === 422
      ) {
        toast.error(
          error?.response?.data?.detail ||
          "Invalid camera information"
        );
      } else if (
        status === 401 ||
        status === 403
      ) {
        toast.error(
          "You are not authorized to add this camera"
        );
      } else if (
        status === 409
      ) {
        toast.error(
          "A camera with this ID already exists"
        );
      } else {
        toast.error(
          error?.response?.data?.detail ||
          "Failed to save camera"
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="space-y-4">

      {/* ==================================================== */}
      {/* SOURCE FORMAT */}
      {/* ==================================================== */}

      <div className="bg-[#0d2038] border border-[#183b63] rounded-md px-3 py-3 text-xs text-[#6ea8d8]">
        <p className="text-lg font-medium">
          <span className="text-2xl font-bold text-blue-400">
            Source Format:
          </span>

          <span className="ml-2">
            {getSourcePlaceholder()}
          </span>
        </p>

        <p className="mt-2 text-sm text-[#779abb]">
          Enter the complete camera/video stream URL.
          The system will not modify or append anything
          to the URL.
        </p>
      </div>

      {/* ==================================================== */}
      {/* CAMERA NAME */}
      {/* ==================================================== */}

      <div>
        <label
          htmlFor="camera-name"
          className="block text-lg font-bold text-[#86bde8] mb-2"
        >
          CAMERA NAME
        </label>

        <input
          id="camera-name"
          type="text"
          value={cameraName}
          onChange={(event) =>
            setCameraName(
              event.target.value
            )
          }
          placeholder="e.g. Gold Shop Entrance CAM-01"
          maxLength={
            MAX_CAMERA_NAME_LENGTH
          }
          autoComplete="off"
          disabled={loading}
          className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none focus:border-blue-500 disabled:opacity-60"
        />
      </div>

      {/* ==================================================== */}
      {/* SOURCE TYPE */}
      {/* ==================================================== */}

      <div>
        <label
          htmlFor="source-type"
          className="block text-lg font-bold text-[#86bde8] mb-2"
        >
          SOURCE TYPE
        </label>

        <select
          id="source-type"
          value={sourceType}
          onChange={
            handleSourceTypeChange
          }
          disabled={loading}
          className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none focus:border-blue-500 text-white disabled:opacity-60"
        >
          <option
            value={SOURCE_TYPES.RTSP}
          >
            RTSP Camera
          </option>

          <option
            value={SOURCE_TYPES.HTTP}
          >
            HTTP / MJPEG Stream
          </option>

          <option
            value={SOURCE_TYPES.HTTPS}
          >
            HTTPS Video / Stream
          </option>

          <option
            value={SOURCE_TYPES.HLS}
          >
            HLS (.m3u8) Stream
          </option>

          <option
            value={SOURCE_TYPES.WEBCAM}
          >
            Local Webcam
          </option>
        </select>
      </div>

      {/* ==================================================== */}
      {/* CAMERA / VIDEO SOURCE */}
      {/* ==================================================== */}

      <div>
        <label
          htmlFor="camera-source"
          className="block text-lg font-bold text-[#86bde8] mb-2"
        >
          CAMERA / VIDEO SOURCE
        </label>

        <input
          id="camera-source"
          type="text"
          value={source}
          onChange={(event) =>
            setSource(
              event.target.value
            )
          }
          placeholder={
            getSourcePlaceholder()
          }
          maxLength={
            MAX_SOURCE_LENGTH
          }
          autoComplete="off"
          spellCheck={false}
          inputMode="url"
          disabled={loading}
          className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none focus:border-blue-500 disabled:opacity-60"
        />

        <p className="text-xs text-[#6687a8] mt-1">
          For RTSP cameras, enter the complete
          RTSP URL including the required channel
          or stream path.
        </p>
      </div>

      {/* ==================================================== */}
      {/* CAMERA LOCATION */}
      {/* ==================================================== */}

      <div>
        <label
          htmlFor="camera-location-name"
          className="block text-lg font-bold text-[#86bde8] mb-2"
        >
          CAMERA LOCATION
        </label>

        <input
          id="camera-location-name"
          type="text"
          value={locationName}
          onChange={(event) =>
            setLocationName(
              event.target.value
            )
          }
          placeholder="e.g. Main Gate, SG Highway"
          maxLength={
            MAX_LOCATION_NAME_LENGTH
          }
          autoComplete="street-address"
          disabled={loading}
          className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none focus:border-blue-500 disabled:opacity-60"
        />

        <p className="text-xs text-[#6687a8] mt-1">
          Enter a recognizable physical location
          for this CCTV camera.
        </p>
      </div>

      {/* ==================================================== */}
      {/* GIS CAMERA LOCATION */}
      {/* ==================================================== */}

      <div className="bg-[#0b1d32] border border-[#21456d] rounded-md p-3">

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-3">

          <div>
            <h3 className="text-xl font-bold text-blue-400">
              GIS CAMERA LOCATION
            </h3>

            <p className="text-sm text-[#86a9c9] mt-1">
              Click on the map to select the physical
              location of this CCTV camera.
            </p>
          </div>

          {selectedLocation && (
            <div className="bg-[#071526] border border-blue-500/50 rounded-md px-3 py-2">

              <p className="text-xs text-gray-400">
                SELECTED LOCATION
              </p>

              <p className="text-sm font-bold text-white">
                {selectedLocation.latitude}
                {" , "}
                {selectedLocation.longitude}
              </p>

            </div>
          )}

        </div>

        <GISMap
          cameras={[]}
          selectedCamera={null}
          onCameraClick={undefined}
          onMapClick={
            loading
              ? undefined
              : handleMapClick
          }
        />

      </div>

      {/* ==================================================== */}
      {/* SELECTED GIS COORDINATES */}
      {/* ==================================================== */}

      <div className="bg-[#0b1d32] border border-[#21456d] rounded-md p-3">

        <div className="mb-3">
          <h3 className="text-xl font-bold text-blue-400">
            SELECTED GIS COORDINATES
          </h3>

          <p className="text-sm text-[#86a9c9] mt-1">
            Coordinates are controlled by the GIS map
            selection and are sent to the backend for
            validation and storage.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

          <div>
            <label
              htmlFor="camera-latitude"
              className="block text-lg font-bold text-[#86bde8] mb-2"
            >
              LATITUDE
            </label>

            <input
              id="camera-latitude"
              type="text"
              value={latitude}
              readOnly
              aria-readonly="true"
              placeholder="Select a point on the GIS map"
              className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none text-white cursor-not-allowed opacity-90"
            />
          </div>

          <div>
            <label
              htmlFor="camera-longitude"
              className="block text-lg font-bold text-[#86bde8] mb-2"
            >
              LONGITUDE
            </label>

            <input
              id="camera-longitude"
              type="text"
              value={longitude}
              readOnly
              aria-readonly="true"
              placeholder="Select a point on the GIS map"
              className="w-full bg-[#142b46] border border-[#21456d] rounded-md px-3 py-3 text-md outline-none text-white cursor-not-allowed opacity-90"
            />
          </div>

        </div>

      </div>

      {/* ==================================================== */}
      {/* SAVE CAMERA */}
      {/* ==================================================== */}

      <button
        type="button"
        onClick={saveCamera}
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 transition rounded-md py-3 font-bold text-xl disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {loading
          ? "Saving..."
          : "Save Camera"}
      </button>

    </div>
  );
};

export default LiveCam;
