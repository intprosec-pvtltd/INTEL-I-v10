import React, { useEffect, useMemo, useRef, useState } from "react";
import toast from "react-hot-toast";
import { Upload, Trash2, MapPin, LocateFixed } from "lucide-react";
import {
  CircleMarker,
  MapContainer,
  TileLayer,
  useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { get, set, del } from "idb-keyval";
import api from "../api/axios";
import Alert from "./Alert";

const STORAGE_KEY = "multiUploadVideos";
const MAX_UPLOAD_VIDEOS = 60;
const MAX_FILE_SIZE_MB = 500;
const DEFAULT_MAP_CENTER = [23.0225, 72.5714];

const UploadMapClickHandler = ({ onSelect }) => {
  useMapEvents({
    click(event) {
      onSelect(event.latlng.lat, event.latlng.lng);
    },
  });
  return null;
};

const UploadLocationPicker = ({ latitude, longitude, onSelect }) => {
  const hasCoordinates =
    String(latitude ?? "").trim() !== "" &&
    String(longitude ?? "").trim() !== "";
  const position =
    hasCoordinates &&
    Number.isFinite(Number(latitude)) &&
    Number.isFinite(Number(longitude))
      ? [Number(latitude), Number(longitude)]
      : null;

  return (
    <div className="relative isolate h-[260px] w-full min-w-0 max-w-full overflow-hidden rounded-lg bg-[#071524] [&_.leaflet-container]:!relative [&_.leaflet-container]:!z-0 [&_.leaflet-container]:!h-full [&_.leaflet-container]:!w-full [&_.leaflet-container]:!max-w-full [&_.leaflet-top]:!z-[20] [&_.leaflet-bottom]:!z-[20]">
      <MapContainer
        center={position || DEFAULT_MAP_CENTER}
        zoom={position ? 15 : 11}
        scrollWheelZoom
        className="!relative !z-0 !h-full !w-full !max-w-full"
      >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <UploadMapClickHandler onSelect={onSelect} />
      {position && (
        <CircleMarker
          center={position}
          radius={9}
          pathOptions={{
            color: "#ffffff",
            weight: 3,
            fillColor: "#06b6d4",
            fillOpacity: 1,
          }}
        />
      )}
      </MapContainer>
    </div>
  );
};

const UploadVideo = () => {
  const [videos, setVideos] = useState([]);
  const [restoring, setRestoring] = useState(true);
  const fileInputRef = useRef(null);

  const getBaseURL = () => {
    return api.defaults.baseURL || import.meta.env.VITE_BASE_URL || "";
  };

  const makePreviewUrl = (file) => URL.createObjectURL(file);

  const getErrorMessage = (error, fallback) => {
    const detail = error?.response?.data?.detail;
    const message = error?.response?.data?.message;

    if (typeof detail?.message === "string") {
      return detail.message;
    }

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      const validationMessage = detail
        .map((entry) => entry?.msg)
        .filter(Boolean)
        .join("; ");

      if (validationMessage) {
        return validationMessage;
      }
    }

    if (typeof message === "string") {
      return message;
    }

    return fallback;
  };

  // Axios instances often have application/json configured as a default.
  // A FormData upload must not reuse that header: the browser must generate
  // multipart/form-data together with its unique boundary parameter.
  const createMultipartUploadConfig = () => ({
    timeout: 10 * 60 * 1000,
    headers: {
      "Content-Type": undefined,
    },
    transformRequest: [
      (data, headers) => {
        if (headers) {
          if (typeof headers.delete === "function") {
            headers.delete("Content-Type");
          } else {
            delete headers["Content-Type"];
            delete headers["content-type"];
          }
        }

        return data;
      },
    ],
  });

  const getSafeUploadFileName = (item) => {
    const candidate = String(
      item?.fileName || item?.file?.name || "upload-video.mp4"
    )
      .normalize("NFKC")
      .replace(/[\r\n"]/g, "_")
      .trim();

    return candidate || "upload-video.mp4";
  };

  // A restored browser entry can outlive its backend Camera row. In that
  // situation DELETE correctly returns 404/410, but the stale local card must
  // still be removed from IndexedDB and from the UI.
  const isAlreadyDeletedError = (error) =>
    error?.response?.status === 404 || error?.response?.status === 410;

  const createUploadStreamUrl = async (streamID) => {
    if (!streamID) {
      throw new Error("Upload stream ID missing");
    }

    await api.post(
      `/stream-session/upload/${encodeURIComponent(streamID)}`
    );

    const baseURL = getBaseURL();

    return `${baseURL}/stream-upload/${encodeURIComponent(
      streamID
    )}?t=${Date.now()}`;
  };

  const persistVideos = async (nextVideos) => {
    const safeVideos = nextVideos.map((item) => ({
      id: item.id,
      file: item.file,
      fileName: item.fileName,
      fileSize: item.fileSize,
      fileType: item.fileType,
      streamID: item.streamID || "",
      camID: item.camID || "",
      streamOn: item.streamOn || false,
      uploaded: item.uploaded || false,
      cameraName: item.cameraName || "",
      locationName: item.locationName || "",
      latitude: item.latitude ?? "",
      longitude: item.longitude ?? "",
    }));

    await set(STORAGE_KEY, safeVideos);
  };

  const updateVideos = async (updater) => {
    setVideos((prev) => {
      const next =
        typeof updater === "function"
          ? updater(prev)
          : updater;

      persistVideos(next).catch((err) =>
        console.error(
          "Persist upload videos error:",
          err
        )
      );

      return next;
    });
  };

  useEffect(() => {
    const restoreVideos = async () => {
      try {
        const savedVideos =
          (await get(STORAGE_KEY)) || [];

        const restored = savedVideos
          .filter((item) => item.file)
          .map((item) => ({
            ...item,
            previewUrl: makePreviewUrl(
              item.file
            ),
            videoURL: "",
            loading: false,
            streamLoading: false,
            stopping: false,
          }));

        setVideos(restored);

        for (const item of restored) {
          if (
            item.streamOn &&
            item.streamID
          ) {
            try {
              const secureUrl =
                await createUploadStreamUrl(
                  item.streamID
                );

              setVideos((prev) =>
                prev.map((v) =>
                  v.id === item.id
                    ? {
                        ...v,
                        videoURL: secureUrl,
                        streamLoading: true,
                      }
                    : v
                )
              );
            } catch (error) {
              console.error(
                "Restore secure upload stream failed:",
                error
              );

              setVideos((prev) =>
                prev.map((v) =>
                  v.id === item.id
                    ? {
                        ...v,
                        streamOn: false,
                        videoURL: "",
                        streamLoading: false,
                      }
                    : v
                )
              );
            }
          }
        }
      } catch (error) {
        console.error(
          "Restore upload videos error:",
          error
        );

        toast.error(
          "Failed to restore uploaded videos"
        );
      } finally {
        setRestoring(false);
      }
    };

    restoreVideos();

    return () => {
      setVideos((prev) => {
        prev.forEach((item) => {
          if (
            item.previewUrl?.startsWith(
              "blob:"
            )
          ) {
            URL.revokeObjectURL(
              item.previewUrl
            );
          }
        });

        return prev;
      });
    };
  }, []);

  const validateFile = (file) => {
    if (
      !file?.type?.startsWith(
        "video/"
      )
    ) {
      toast.error(
        `${
          file?.name || "Selected file"
        } is not a video file`
      );

      return false;
    }

    const sizeMB =
      file.size /
      (1024 * 1024);

    if (
      sizeMB >
      MAX_FILE_SIZE_MB
    ) {
      toast.error(
        `${file.name} is larger than ${MAX_FILE_SIZE_MB}MB`
      );

      return false;
    }

    return true;
  };

  const handleVideoChange = async (e) => {
    const selectedFiles =
      Array.from(
        e.target.files || []
      );

    if (
      selectedFiles.length === 0
    ) {
      toast.error(
        "No file selected"
      );

      return;
    }

    const validFiles =
      selectedFiles.filter(
        validateFile
      );

    if (
      validFiles.length === 0
    ) {
      return;
    }

    if (
      videos.length +
        validFiles.length >
      MAX_UPLOAD_VIDEOS
    ) {
      toast.error(
        `Maximum ${MAX_UPLOAD_VIDEOS} videos allowed at a time`
      );

      return;
    }

    const newItems =
      validFiles.map(
        (file, fileIndex) => ({
          id: crypto.randomUUID(),
          file,
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          previewUrl:
            makePreviewUrl(file),
          streamID: "",
          camID: "",
          videoURL: "",
          streamOn: false,
          uploaded: false,
          loading: false,
          streamLoading: false,
          stopping: false,
          cameraName: `Upload Camera ${videos.length + fileIndex + 1}`,
          locationName: "",
          latitude: "",
          longitude: "",
        })
      );

    await updateVideos(
      (prev) => [
        ...prev,
        ...newItems,
      ]
    );

    if (
      fileInputRef.current
    ) {
      fileInputRef.current.value =
        "";
    }

    toast.success(
      `${newItems.length} video added`
    );
  };

  const updateVideoField = (itemId, field, value) => {
    updateVideos((prev) =>
      prev.map((video) =>
        video.id === itemId ? { ...video, [field]: value } : video
      )
    );
  };

  const setVideoLocation = (itemId, latitude, longitude) => {
    updateVideos((prev) =>
      prev.map((video) =>
        video.id === itemId
          ? {
              ...video,
              latitude: Number(latitude).toFixed(6),
              longitude: Number(longitude).toFixed(6),
            }
          : video
      )
    );
  };

  const selectCurrentLocation = (itemId) => {
    if (!navigator.geolocation) {
      toast.error("Geolocation is not supported by this browser");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        setVideoLocation(itemId, coords.latitude, coords.longitude);
        toast.success("Current location selected");
      },
      () => toast.error("Unable to read current location"),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  const uploadVideo = async (
    itemId
  ) => {
    const item = videos.find(
      (v) => v.id === itemId
    );

    if (!item?.file) {
      toast.error(
        "Video file missing"
      );

      return;
    }

    if (!(item.file instanceof Blob) || item.file.size <= 0) {
      toast.error(
        "The stored video is empty or no longer readable. Remove it and select the file again."
      );
      return;
    }

    const hasCoordinateValues =
      String(item.latitude ?? "").trim() !== "" &&
      String(item.longitude ?? "").trim() !== "";
    const latitude = Number(item.latitude);
    const longitude = Number(item.longitude);
    if (
      !hasCoordinateValues ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude) ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      toast.error("Select a valid GIS location on the map before uploading");
      return;
    }

    try {
      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  loading: true,
                }
              : v
          )
      );

      const formData =
        new FormData();

      formData.append(
        "videoFile",
        item.file,
        getSafeUploadFileName(item)
      );
      formData.append("camera_name", item.cameraName.trim() || item.fileName);
      formData.append("location_name", item.locationName.trim());
      formData.append("latitude", String(latitude));
      formData.append("longitude", String(longitude));

      const res = await api.post(
        "/video-upload",
        formData,
        createMultipartUploadConfig()
      );

      const newStreamID =
        res.data?.streamID;

      const newCamID =
        res.data?.camID;

      if (
        !newStreamID ||
        !newCamID
      ) {
        toast.error(
          "Stream ID or Camera ID not received from backend"
        );

        await updateVideos(
          (prev) =>
            prev.map((v) =>
              v.id === itemId
                ? {
                    ...v,
                    loading: false,
                  }
                : v
            )
        );

        return;
      }

      const secureUrl =
        await createUploadStreamUrl(
          newStreamID
        );

      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  streamID:
                    newStreamID,
                  camID:
                    newCamID,
                  videoURL:
                    secureUrl,
                  streamOn: true,
                  streamLoading:
                    true,
                  uploaded: true,
                  loading: false,
                }
              : v
          )
      );

      toast.success(
        `${item.fileName} uploaded and analysis started`
      );
    } catch (error) {
      console.error(
        "Upload error:",
        error
      );

      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  loading: false,
                }
              : v
          )
      );

      if (
        error.code ===
          "ECONNABORTED" ||
        String(
          error.message || ""
        )
          .toLowerCase()
          .includes("timeout")
      ) {
        toast.error(
          "Upload timed out. Try a smaller video or check the network connection."
        );
      } else if (
        error.response?.status ===
        429
      ) {
        toast.error(
          error.response?.data
            ?.detail?.message ||
            "System limit reached. Stop another stream first."
        );
      } else if (
        [400, 422].includes(error.response?.status) &&
        getErrorMessage(error, "")
          .toLowerCase()
          .includes("parsing the body")
      ) {
        toast.error(
          "The backend could not parse the multipart upload. Refresh the page, select the video again, and verify that no Axios interceptor forces Content-Type to application/json."
        );
      } else {
        toast.error(
          getErrorMessage(
            error,
            "Video upload failed"
          )
        );
      }
    }
  };

  const stopVideo = async (
    itemId
  ) => {
    const item = videos.find(
      (v) => v.id === itemId
    );

    try {
      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  stopping: true,
                }
              : v
          )
      );

      if (item?.camID) {
        await api.post(
          `/camera/${encodeURIComponent(
            item.camID
          )}/stop`,
          {}
        );
      }

      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  streamOn: false,
                  videoURL: "",
                  streamLoading: false,
                  stopping: false,
                }
              : v
          )
      );

      toast.success(
        "Video stream stopped"
      );
    } catch (error) {
      console.error(
        "Stop video error:",
        error
      );

      await updateVideos(
        (prev) =>
          prev.map((v) =>
            v.id === itemId
              ? {
                  ...v,
                  stopping: false,
                }
              : v
          )
      );

      toast.error(
        getErrorMessage(
          error,
          "Failed to stop video"
        )
      );
    }
  };

  const removeVideo = async (
    itemId
  ) => {
    const item = videos.find(
      (v) => v.id === itemId
    );

    if (!item) return;

    try {
      await updateVideos((prev) =>
        prev.map((video) =>
          video.id === itemId
            ? { ...video, stopping: true }
            : video
        )
      );

      // The backend delete endpoint stops the worker and removes the upload,
      // Camera row, Redis/runtime state and GIS cache in one operation.
      if (item.camID) {
        try {
          await api.delete(
            `/cameras/${encodeURIComponent(item.camID)}`
          );
        } catch (error) {
          // "Camera not found" means the database/backend cleanup has already
          // happened. Continue and purge the stale browser copy.
          if (!isAlreadyDeletedError(error)) {
            throw error;
          }

          console.info(
            `Camera ${item.camID} was already deleted; removing stale local entry.`
          );
        }
      }

      if (item.previewUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(item.previewUrl);
      }

      await updateVideos((prev) =>
        prev.filter((video) => video.id !== itemId)
      );

      toast.success(
        item.camID
          ? "Video and GIS camera deleted"
          : "Video removed"
      );
    } catch (error) {
      console.error("Delete uploaded video error:", error);
      await updateVideos((prev) =>
        prev.map((video) =>
          video.id === itemId
            ? { ...video, stopping: false }
            : video
        )
      );
      toast.error(
        getErrorMessage(error, "Failed to delete video and GIS camera")
      );
    }
  };

  const clearAll = async () => {
    try {
      const failedIds = new Set();

      for (const item of videos) {
        if (item.camID) {
          try {
            await api.delete(
              `/cameras/${encodeURIComponent(
                item.camID
              )}`
            );
          } catch (error) {
            // Do not keep a stale browser card when the backend confirms that
            // the camera has already gone.
            if (isAlreadyDeletedError(error)) {
              console.info(
                `Camera ${item.camID} was already deleted; clearing stale local entry.`
              );
            } else {
              console.error(
                "Delete during clear failed:",
                error
              );
              failedIds.add(item.id);
            }
          }
        }

        if (!failedIds.has(item.id) &&
          item.previewUrl?.startsWith(
            "blob:"
          )
        ) {
          URL.revokeObjectURL(
            item.previewUrl
          );
        }
      }

      if (failedIds.size === 0) {
        await del(STORAGE_KEY);
        setVideos([]);
        toast.success("All videos and GIS cameras deleted");
      } else {
        const remaining = videos.filter((item) => failedIds.has(item.id));
        await persistVideos(remaining);
        setVideos(remaining);
        toast.error(`${failedIds.size} video(s) could not be deleted`);
      }
    } catch (error) {
      console.error(
        "Clear all error:",
        error
      );

      toast.error(
        "Failed to clear videos"
      );
    }
  };

  const refreshUploadStream =
    async (itemId) => {
      const item = videos.find(
        (v) => v.id === itemId
      );

      if (!item?.streamID) {
        return;
      }

      try {
        await updateVideos(
          (prev) =>
            prev.map((v) =>
              v.id === itemId
                ? {
                    ...v,
                    streamLoading:
                      true,
                  }
                : v
            )
        );

        const secureUrl =
          await createUploadStreamUrl(
            item.streamID
          );

        await updateVideos(
          (prev) =>
            prev.map((v) =>
              v.id === itemId
                ? {
                    ...v,
                    videoURL:
                      secureUrl,
                    streamOn: true,
                    streamLoading:
                      true,
                  }
                : v
            )
        );
      } catch (error) {
        console.error(
          "Refresh upload stream error:",
          error
        );

        await updateVideos(
          (prev) =>
            prev.map((v) =>
              v.id === itemId
                ? {
                    ...v,
                    streamLoading:
                      false,
                  }
                : v
            )
        );

        toast.error(
          "Failed to refresh upload stream session"
        );
      }
    };

  const activeUploadVideos =
    useMemo(() => {
      return videos.filter(
        (item) =>
          Boolean(item.camID)
      );
    }, [videos]);

  const shouldShowUploadAlerts =
    activeUploadVideos.length >
    0;

  const uploadVideoMap =
    useMemo(() => {
      return videos.reduce(
        (acc, item, index) => {
          if (item.camID) {
            acc[item.camID] = {
              videoNumber:
                index + 1,
              fileName:
                item.fileName,
            };
          }

          return acc;
        },
        {}
      );
    }, [videos]);

  if (restoring) {
    return (
      <div className="bg-[#0d2038] border border-[#183b63] rounded-md px-4 py-4 text-[#86bde8]">
        Restoring uploaded videos...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-[#0d2038] border border-[#183b63] rounded-md px-3 py-3 text-xs text-[#6ea8d8]">
        <p className="text-lg font-medium">
          <span className="text-2xl font-bold text-blue-400">
            Supported :{" "}
          </span>

          MP4, AVI, MOV, MKV,
          WebM, FLV. Multiple
          uploaded videos can run
          at the same time based on
          system capacity.
        </p>
      </div>

      <div>
        <label className="block text-xl font-bold tracking-wider text-[#86bde8] mb-2">
          VIDEO FILES
        </label>

        <input
          ref={fileInputRef}
          id="videoUploadInput"
          type="file"
          accept="video/*"
          multiple
          onChange={
            handleVideoChange
          }
          className="hidden"
        />

        <label
          htmlFor="videoUploadInput"
          className="h-[180px] flex flex-col items-center justify-center rounded-lg bg-[#142b46] border border-dashed border-[#21456d] text-[#5b82a8] cursor-pointer"
        >
          <Upload
            className="mb-2"
            size={62}
          />

          <span className="text-lg font-medium">
            Click to browse one or
            more videos
          </span>

          <span className="text-sm mt-1">
            Max {MAX_UPLOAD_VIDEOS}{" "}
            videos,{" "}
            {MAX_FILE_SIZE_MB}MB each
          </span>
        </label>
      </div>

      {videos.length > 0 && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={clearAll}
            className="bg-red-700 hover:bg-red-800 px-4 py-2 rounded-md font-bold text-white"
          >
            Clear All
          </button>
        </div>
      )}

      <div
        className={`grid min-w-0 grid-cols-1 gap-5 ${
          shouldShowUploadAlerts
            ? "2xl:grid-cols-[1fr_430px]"
            : ""
        }`}
      >
        <div className="grid min-w-0 grid-cols-1 xl:grid-cols-2 gap-5">
          {videos.map(
            (item, index) => (
              <div
                key={item.id}
                className="min-w-0 max-w-full overflow-hidden bg-[#0d2038] border border-[#183b63] rounded-xl p-4 space-y-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-blue-400 font-bold text-lg break-all">
                      Video {index + 1}:{" "}
                      {item.fileName}
                    </h3>

                    <p className="text-sm text-gray-300">
                      Size:{" "}
                      {(
                        item.fileSize /
                        (1024 * 1024)
                      ).toFixed(2)}{" "}
                      MB
                    </p>

                    {item.camID && (
                      <p className="text-xs text-cyan-300 break-all mt-1">
                        Camera ID:{" "}
                        {item.camID}
                      </p>
                    )}

                    <p
                      className={`text-sm font-bold mt-1 ${
                        item.streamOn
                          ? "text-green-400"
                          : "text-yellow-400"
                      }`}
                    >
                      {item.streamOn
                        ? "Analysing"
                        : item.uploaded
                        ? "Stopped"
                        : "Ready to upload"}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      removeVideo(
                        item.id
                      )
                    }
                    disabled={
                      item.loading ||
                      item.stopping
                    }
                    className="bg-red-600 hover:bg-red-700 disabled:opacity-60 p-2 rounded-md"
                  >
                    <Trash2
                      size={18}
                    />
                  </button>
                </div>

                <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-[#21456d] bg-[#091a2d] p-4 space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <MapPin size={20} className="text-cyan-400" />
                      <div>
                        <h4 className="font-bold text-cyan-300">
                          GIS Camera Location
                        </h4>
                        <p className="text-xs text-[#6ea8d8]">
                          Select where this uploaded video was recorded.
                        </p>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => selectCurrentLocation(item.id)}
                      disabled={item.uploaded || item.loading}
                      className="flex items-center gap-2 rounded-md border border-cyan-700 bg-cyan-950/60 px-3 py-2 text-xs font-bold text-cyan-300 hover:bg-cyan-900/60 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <LocateFixed size={16} />
                      Use Current Location
                    </button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <label className="space-y-1">
                      <span className="text-xs font-bold tracking-wide text-[#86bde8]">
                        CAMERA NAME
                      </span>
                      <input
                        type="text"
                        value={item.cameraName || ""}
                        onChange={(event) =>
                          updateVideoField(item.id, "cameraName", event.target.value)
                        }
                        disabled={item.uploaded || item.loading}
                        maxLength={200}
                        placeholder="Upload Camera A"
                        className="w-full rounded-md border border-[#21456d] bg-[#071524] px-3 py-2 text-white outline-none focus:border-cyan-500 disabled:opacity-60"
                      />
                    </label>

                    <label className="space-y-1">
                      <span className="text-xs font-bold tracking-wide text-[#86bde8]">
                        LOCATION NAME
                      </span>
                      <input
                        type="text"
                        value={item.locationName || ""}
                        onChange={(event) =>
                          updateVideoField(item.id, "locationName", event.target.value)
                        }
                        disabled={item.uploaded || item.loading}
                        maxLength={255}
                        placeholder="Ahmedabad Point A"
                        className="w-full rounded-md border border-[#21456d] bg-[#071524] px-3 py-2 text-white outline-none focus:border-cyan-500 disabled:opacity-60"
                      />
                    </label>

                    <label className="space-y-1">
                      <span className="text-xs font-bold tracking-wide text-[#86bde8]">
                        LATITUDE
                      </span>
                      <input
                        type="number"
                        step="any"
                        min="-90"
                        max="90"
                        value={item.latitude ?? ""}
                        onChange={(event) =>
                          updateVideoField(item.id, "latitude", event.target.value)
                        }
                        disabled={item.uploaded || item.loading}
                        placeholder="23.022500"
                        className="w-full rounded-md border border-[#21456d] bg-[#071524] px-3 py-2 text-white outline-none focus:border-cyan-500 disabled:opacity-60"
                      />
                    </label>

                    <label className="space-y-1">
                      <span className="text-xs font-bold tracking-wide text-[#86bde8]">
                        LONGITUDE
                      </span>
                      <input
                        type="number"
                        step="any"
                        min="-180"
                        max="180"
                        value={item.longitude ?? ""}
                        onChange={(event) =>
                          updateVideoField(item.id, "longitude", event.target.value)
                        }
                        disabled={item.uploaded || item.loading}
                        placeholder="72.571400"
                        className="w-full rounded-md border border-[#21456d] bg-[#071524] px-3 py-2 text-white outline-none focus:border-cyan-500 disabled:opacity-60"
                      />
                    </label>
                  </div>

                  <div className="relative isolate w-full min-w-0 max-w-full overflow-hidden rounded-lg border border-[#21456d]">
                    <UploadLocationPicker
                      latitude={item.latitude}
                      longitude={item.longitude}
                      onSelect={(latitude, longitude) =>
                        !item.uploaded &&
                        !item.loading &&
                        setVideoLocation(item.id, latitude, longitude)
                      }
                    />
                  </div>

                  <p className="text-xs text-[#6ea8d8]">
                    Click the map to place the camera. Use a different location for each uploaded video to demonstrate cross-camera movement.
                  </p>
                </div>

                <div className="w-full h-[420px] rounded-lg border border-[#21456d] bg-black overflow-hidden flex items-center justify-center">
                  {item.streamOn &&
                  item.videoURL ? (
                    <div className="relative w-full h-full flex items-center justify-center bg-black">
                      {item.streamLoading && (
                        <div
                          className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/70 backdrop-blur-[1px]"
                          role="status"
                          aria-live="polite"
                          aria-label="Loading video stream"
                        >
                          <div
                            className="h-12 w-12 rounded-full border-4 border-[#21456d] border-t-blue-400 animate-spin"
                            aria-hidden="true"
                          />

                          <span className="mt-3 text-sm font-semibold text-blue-200">
                            Loading video stream...
                          </span>
                        </div>
                      )}

                      <img
                        key={
                          item.videoURL
                        }
                        src={
                          item.videoURL
                        }
                        alt={`Live analysis stream for ${item.fileName}`}
                        className="w-full h-full object-contain"
                        decoding="async"
                        onLoad={() => {
                          setVideos(
                            (prev) =>
                              prev.map(
                                (v) =>
                                  v.id ===
                                  item.id
                                    ? {
                                        ...v,
                                        streamLoading:
                                          false,
                                      }
                                    : v
                              )
                          );
                        }}
                        onError={() => {
                          setVideos(
                            (prev) =>
                              prev.map(
                                (v) =>
                                  v.id ===
                                  item.id
                                    ? {
                                        ...v,
                                        streamLoading:
                                          true,
                                      }
                                    : v
                              )
                          );

                          toast.error(
                            "Upload secure stream session expired. Reconnecting..."
                          );

                          refreshUploadStream(
                            item.id
                          );
                        }}
                      />
                    </div>
                  ) : (
                    <video
                      src={
                        item.previewUrl
                      }
                      controls
                      className="w-full h-full object-contain"
                    />
                  )}
                </div>

                {!item.streamOn ? (
                  <button
                    type="button"
                    onClick={() =>
                      uploadVideo(
                        item.id
                      )
                    }
                    disabled={
                      item.loading
                    }
                    className="w-full bg-blue-600 text-xl hover:bg-blue-700 transition rounded-md py-3 font-bold disabled:opacity-60"
                  >
                    {item.loading
                      ? "Uploading..."
                      : "Upload & Analyse"}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() =>
                      stopVideo(
                        item.id
                      )
                    }
                    disabled={
                      item.stopping
                    }
                    className="w-full bg-red-600 text-xl hover:bg-red-700 transition rounded-md py-3 font-bold disabled:opacity-60"
                  >
                    {item.stopping
                      ? "Stopping..."
                      : "Stop Video Stream"}
                  </button>
                )}
              </div>
            )
          )}
        </div>

        {shouldShowUploadAlerts && (
          <aside
            className="min-h-[400px] self-start rounded-xl border border-[#183b63] bg-[#07182b] p-4 2xl:sticky 2xl:top-[136px] 2xl:flex 2xl:h-[calc(100dvh-152px)] 2xl:min-h-0 2xl:flex-col 2xl:overflow-hidden"
          >
            <Alert
              sourceMode="upload"
              allowedTypes={[
                "upload",
              ]}
              uploadVideoMap={
                uploadVideoMap
              }
            />
          </aside>
        )}
      </div>
    </div>
  );
};

export default UploadVideo;
