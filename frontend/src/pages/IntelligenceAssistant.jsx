import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  Bot,
  Send,
  ShieldCheck,
  LoaderCircle,
  ExternalLink,
  Trash2,
} from "lucide-react";

import {
  useLocation,
} from "react-router-dom";

import api from "../api/axios.js";
import { formatIndianDateTime } from "../utils/dateTime.js";

import {
  useNavigationStack,
} from "../context/NavigationStackContext";

/* =========================================================
   DEFAULT SUGGESTIONS
========================================================= */

const DEFAULT_SUGGESTIONS = [
  "What is happening now?",
  "Show newest alerts",
  "Show high-priority alerts",
  "Which cameras are offline?",
  "What happened in the last hour?",
  "Show recent watchlist matches",
  "Is the AI system healthy?",
];

/* =========================================================
   STORAGE CONFIGURATION
========================================================= */

const CHAT_STORAGE_PREFIX =
  "intel_i_intelligence_assistant_chat_v2";

const SCROLL_STORAGE_PREFIX =
  "intel_i_intelligence_assistant_scroll_v2";

const MAX_STORED_MESSAGES = 200;

/* =========================================================
   MESSAGE ID
========================================================= */

const createMessageId = () => {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random()
      .toString(36)
      .slice(2)}`;
  }
};

/* =========================================================
   CURRENT USER IDENTITY
========================================================= */

const getCurrentUserIdentity = () => {
  try {
    const rawUser =
      localStorage.getItem("user");

    if (!rawUser) {
      return "session";
    }

    const user =
      JSON.parse(rawUser);

    return String(
      user?.id ||
        user?.email ||
        user?.username ||
        "session",
    );
  } catch (error) {
    console.error(
      "Unable to resolve INTEL-I assistant user:",
      error,
    );

    return "session";
  }
};

/* =========================================================
   STORAGE KEYS
========================================================= */

const getChatStorageKey = () => {
  return `${CHAT_STORAGE_PREFIX}:${getCurrentUserIdentity()}`;
};

const getScrollStorageKey = () => {
  return `${SCROLL_STORAGE_PREFIX}:${getCurrentUserIdentity()}`;
};

/* =========================================================
   LOAD STORED MESSAGES
========================================================= */

const loadStoredMessages = (
  storageKey,
) => {
  try {
    const stored =
      sessionStorage.getItem(
        storageKey,
      );

    if (!stored) {
      return [];
    }

    const parsed =
      JSON.parse(stored);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .filter(
        (message) =>
          message &&
          typeof message === "object" &&
          typeof message.text === "string" &&
          [
            "user",
            "assistant",
            "error",
          ].includes(
            message.role,
          ),
      )
      .map((message) => ({
        ...message,

        id:
          message.id ||
          createMessageId(),
      }))
      .slice(
        -MAX_STORED_MESSAGES,
      );
  } catch (error) {
    console.error(
      "Unable to restore INTEL-I assistant chat:",
      error,
    );

    return [];
  }
};

/* =========================================================
   NORMALIZE REFERENCES
========================================================= */

const normalizeReferences = (
  references,
) => {
  if (!Array.isArray(references)) {
    return [];
  }

  return references
    .filter(
      (reference) =>
        reference &&
        typeof reference ===
          "object",
    )
    .slice(0, 20);
};

/* =========================================================
   SAFE INTERNAL ROUTE
========================================================= */

const getInternalRoute = (
  href,
) => {
  if (
    typeof href !== "string"
  ) {
    return null;
  }

  const route =
    href.trim();

  /*
   * Only allow internal INTEL-I routes.
   */
  if (
    !route.startsWith("/") ||
    route.startsWith("//")
  ) {
    return null;
  }

  return route;
};

/* =========================================================
   ERROR MESSAGE
========================================================= */

const getErrorMessage = (
  error,
) => {
  const detail =
    error?.response?.data?.detail;

  if (
    typeof detail === "string" &&
    detail.trim()
  ) {
    return detail;
  }

  if (
    detail &&
    typeof detail === "object"
  ) {
    if (
      typeof detail.message ===
        "string" &&
      detail.message.trim()
    ) {
      return detail.message;
    }
  }

  const message =
    error?.response?.data?.message;

  if (
    typeof message === "string" &&
    message.trim()
  ) {
    return message;
  }

  return (
    "INTEL-I Intelligence Assistant " +
    "is temporarily unavailable."
  );
};

/* =========================================================
   INTELLIGENCE ASSISTANT
========================================================= */

const IntelligenceAssistant = () => {
  const location =
    useLocation();

  const {
    pushRoute,
  } =
    useNavigationStack();

  /* =======================================================
     STORAGE REFERENCES
  ======================================================= */

  const storageKeyRef =
    useRef(
      getChatStorageKey(),
    );

  const scrollStorageKeyRef =
    useRef(
      getScrollStorageKey(),
    );

  /* =======================================================
     UI STATE
  ======================================================= */

  const [
    question,
    setQuestion,
  ] = useState("");

  const [
    messages,
    setMessages,
  ] = useState(() =>
    loadStoredMessages(
      storageKeyRef.current,
    ),
  );

  const [
    suggestions,
    setSuggestions,
  ] = useState(
    DEFAULT_SUGGESTIONS,
  );

  const [
    loading,
    setLoading,
  ] = useState(false);

  /* =======================================================
     DOM REFERENCES
  ======================================================= */

  const chatScrollRef =
    useRef(null);

  const bottomRef =
    useRef(null);

  const scrollRestoreDoneRef =
    useRef(false);

  /* =======================================================
     LOAD SUGGESTIONS
  ======================================================= */

  useEffect(() => {
    let active = true;

    api
      .get(
        "/api/intelligence-assistant/suggestions",
      )
      .then(({ data }) => {
        if (
          !active ||
          !Array.isArray(
            data?.suggestions,
          )
        ) {
          return;
        }

        const validSuggestions =
          data.suggestions
            .filter(
              (item) =>
                typeof item ===
                  "string" &&
                item.trim(),
            )
            .slice(
              0,
              10,
            );

        if (
          validSuggestions.length >
          0
        ) {
          setSuggestions(
            validSuggestions,
          );
        }
      })
      .catch(() => {
        /*
         * Keep built-in suggestions.
         */
      });

    return () => {
      active = false;
    };
  }, []);

  /* =======================================================
     PERSIST CHAT
  ======================================================= */

  useEffect(() => {
    try {
      sessionStorage.setItem(
        storageKeyRef.current,

        JSON.stringify(
          messages.slice(
            -MAX_STORED_MESSAGES,
          ),
        ),
      );
    } catch (error) {
      console.error(
        "Unable to persist INTEL-I assistant chat:",
        error,
      );
    }
  }, [
    messages,
  ]);

  /* =======================================================
     RESTORE CHAT SCROLL POSITION
  ======================================================= */

  useEffect(() => {
    const frame =
      window.requestAnimationFrame(
        () => {
          const container =
            chatScrollRef.current;

          if (!container) {
            scrollRestoreDoneRef.current =
              true;

            return;
          }

          let savedScroll = null;

          /*
           * Prefer state returned by
           * NavigationStackContext.
           */
          const stateScroll =
            Number(
              location.state
                ?.assistantScrollTop,
            );

          if (
            Number.isFinite(
              stateScroll,
            )
          ) {
            savedScroll =
              stateScroll;
          }

          /*
           * Otherwise use sessionStorage.
           */
          if (
            savedScroll === null
          ) {
            try {
              const storedScroll =
                Number(
                  sessionStorage.getItem(
                    scrollStorageKeyRef.current,
                  ),
                );

              if (
                Number.isFinite(
                  storedScroll,
                )
              ) {
                savedScroll =
                  storedScroll;
              }
            } catch {
              // Ignore.
            }
          }

          if (
            savedScroll !== null &&
            savedScroll >= 0
          ) {
            container.scrollTop =
              savedScroll;
          } else if (
            messages.length >
            0
          ) {
            /*
             * First entry with existing messages:
             * show the latest conversation.
             */
            container.scrollTop =
              container.scrollHeight;
          }

          scrollRestoreDoneRef.current =
            true;
        },
      );

    return () => {
      window.cancelAnimationFrame(
        frame,
      );
    };
  }, []);

  /* =======================================================
     AUTO SCROLL FOR NEW MESSAGES
  ======================================================= */

  useEffect(() => {
    if (
      !scrollRestoreDoneRef.current
    ) {
      return;
    }

    bottomRef.current
      ?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
  }, [
    messages,
    loading,
  ]);

  /* =======================================================
     SAVE CURRENT CHAT SCROLL
  ======================================================= */

  const saveCurrentScroll = () => {
    const scrollTop =
      chatScrollRef.current
        ?.scrollTop || 0;

    try {
      sessionStorage.setItem(
        scrollStorageKeyRef.current,
        String(scrollTop),
      );
    } catch {
      // Navigation should continue.
    }

    return scrollTop;
  };

  const handleChatScroll = () => {
    if (
      !chatScrollRef.current
    ) {
      return;
    }

    try {
      sessionStorage.setItem(
        scrollStorageKeyRef.current,

        String(
          chatScrollRef.current
            .scrollTop,
        ),
      );
    } catch {
      // Ignore storage failure.
    }
  };

  /* =======================================================
     OPEN INTELLIGENCE REFERENCE
  ======================================================= */

  const openReference = (
    route,
    reference,
  ) => {
    if (!route) {
      return;
    }

    const scrollTop =
      saveCurrentScroll();

    /*
     * IMPORTANT:
     *
     * Do not use <Link> here.
     *
     * pushRoute() stores this assistant
     * page in the INTEL-I navigation stack
     * before navigating.
     */
    pushRoute(
      route,
      {
        state: {
          intelIReference: {
            type:
              reference?.type ||
              null,

            id:
              reference?.id ||
              null,
          },

          intelIReferenceSource:
            "intelligence-assistant",

          intelIReferenceTimestamp:
            Date.now(),
        },

        returnState: {
          restoreAssistant:
            true,

          assistantScrollTop:
            scrollTop,
        },
      },
    );
  };

  /* =======================================================
     CLEAR CHAT
  ======================================================= */

  const clearChat = () => {
    if (loading) {
      return;
    }

    setMessages([]);
    setQuestion("");

    try {
      sessionStorage.removeItem(
        storageKeyRef.current,
      );

      sessionStorage.removeItem(
        scrollStorageKeyRef.current,
      );
    } catch (error) {
      console.error(
        "Unable to clear INTEL-I assistant state:",
        error,
      );
    }

    if (
      chatScrollRef.current
    ) {
      chatScrollRef.current.scrollTop =
        0;
    }
  };

  /* =======================================================
     ASK INTEL-I
  ======================================================= */

  const ask = async (
    value = question,
  ) => {
    const text =
      String(
        value || "",
      ).trim();

    if (
      !text ||
      loading
    ) {
      return;
    }

    const userMessage = {
      id:
        createMessageId(),

      role:
        "user",

      text,

      createdAt:
        new Date().toISOString(),
    };

    setQuestion("");

    setMessages(
      (previous) => [
        ...previous,
        userMessage,
      ],
    );

    setLoading(true);

    try {
      const {
        data,
      } =
        await api.post(
          "/api/intelligence-assistant/chat",
          {
            question:
              text,
          },
        );

      const assistantMessage = {
        id:
          createMessageId(),

        role:
          "assistant",

        text:
          String(
            data?.answer ||
              "No answer was returned.",
          ),

        generatedAt:
          data?.generated_at ||
          null,

        references:
          normalizeReferences(
            data?.references,
          ),

        createdAt:
          new Date().toISOString(),
      };

      setMessages(
        (previous) => [
          ...previous,
          assistantMessage,
        ],
      );
    } catch (error) {
      setMessages(
        (previous) => [
          ...previous,

          {
            id:
              createMessageId(),

            role:
              "error",

            text:
              getErrorMessage(
                error,
              ),

            createdAt:
              new Date().toISOString(),
          },
        ],
      );
    } finally {
      setLoading(false);
    }
  };

  /* =======================================================
     UI
  ======================================================= */

  return (
    <div
      className="
        mx-auto
        flex
        h-[calc(100vh-118px)]
        w-full
        max-w-[1500px]
        flex-col
        overflow-hidden
        rounded-2xl
        border
        border-slate-800
        bg-[#06101c]
        shadow-2xl
        shadow-black/30
      "
    >
      {/* ===================================================
          HEADER
      =================================================== */}

      <div
        className="
          flex
          shrink-0
          items-center
          justify-between
          gap-4
          border-b
          border-slate-800
          bg-[#081522]
          px-5
          py-4
        "
      >
        {/* LEFT */}

        <div
          className="
            flex
            min-w-0
            items-center
            gap-3
          "
        >
          <div
            className="
              flex
              h-11
              w-11
              shrink-0
              items-center
              justify-center
              rounded-xl
              bg-blue-600/15
              text-blue-400
            "
          >
            <Bot
              size={24}
            />
          </div>

          <div
            className="
              min-w-0
            "
          >
            <h1
              className="
                truncate
                text-lg
                font-bold
                text-white
              "
            >
              INTEL-I Intelligence
              Assistant
            </h1>

            <p
              className="
                truncate
                text-xs
                text-slate-400
              "
            >
              Grounded in your
              authorized live
              INTEL-I records
            </p>
          </div>
        </div>

        {/* RIGHT */}

        <div
          className="
            flex
            shrink-0
            items-center
            gap-2
          "
        >
          <div
            className="
              hidden
              items-center
              gap-2
              rounded-full
              border
              border-emerald-500/20
              bg-emerald-500/10
              px-3
              py-1.5
              text-xs
              font-semibold
              text-emerald-300
              md:flex
            "
          >
            <ShieldCheck
              size={15}
            />

            Read-only intelligence
          </div>

          <button
            type="button"
            onClick={
              clearChat
            }
            disabled={
              loading ||
              messages.length === 0
            }
            title="Clear chat"
            aria-label="Clear intelligence assistant chat"
            className="
              inline-flex
              h-9
              items-center
              justify-center
              gap-2
              rounded-lg
              border
              border-red-500/20
              bg-red-500/[0.07]
              px-3
              text-xs
              font-semibold
              text-red-300
              transition
              hover:border-red-400/40
              hover:bg-red-500/15
              hover:text-red-200
              disabled:cursor-not-allowed
              disabled:opacity-35
            "
          >
            <Trash2
              size={15}
            />

            <span
              className="
                hidden
                sm:inline
              "
            >
              Clear Chat
            </span>
          </button>
        </div>
      </div>

      {/* ===================================================
          CHAT SCROLL CONTAINER
      =================================================== */}

      <div
        ref={
          chatScrollRef
        }
        onScroll={
          handleChatScroll
        }
        className="
          flex-1
          overflow-y-auto
          px-4
          py-5
          sm:px-6
        "
      >
        {/* =================================================
            EMPTY CHAT
        ================================================= */}

        {messages.length ===
          0 && (
          <div
            className="
              mx-auto
              w-full
              max-w-[1280px]
            "
          >
            <div
              className="
                rounded-2xl
                border
                border-slate-800
                bg-[#0a1725]
                p-5
              "
            >
              <p
                className="
                  font-semibold
                  text-white
                "
              >
                Ask INTEL-I what is
                happening now.
              </p>

              <p
                className="
                  mt-1
                  text-sm
                  leading-6
                  text-slate-400
                "
              >
                Answers are generated
                only from authorized
                alerts, cameras,
                incidents, plate
                observations and
                system-health facts.
                The assistant does not
                invent CCTV events.
              </p>
            </div>

            {/* SUGGESTIONS */}

            <div
              className="
                mt-4
                grid
                gap-2
                sm:grid-cols-2
              "
            >
              {suggestions.map(
                (item) => (
                  <button
                    key={
                      item
                    }
                    type="button"
                    disabled={
                      loading
                    }
                    onClick={() =>
                      ask(item)
                    }
                    className="
                      rounded-xl
                      border
                      border-slate-800
                      bg-[#0a1725]
                      px-4
                      py-3
                      text-left
                      text-sm
                      text-slate-300
                      transition
                      hover:border-blue-500/40
                      hover:bg-[#10233a]
                      hover:text-white
                      disabled:cursor-not-allowed
                      disabled:opacity-50
                    "
                  >
                    {item}
                  </button>
                ),
              )}
            </div>
          </div>
        )}

        {/* =================================================
            MESSAGE LIST
        ================================================= */}

        <div
          className="
            mx-auto
            w-full
            max-w-[1280px]
            space-y-4
          "
        >
          {messages.map(
            (message) => {
              const isUser =
                message.role ===
                "user";

              const isError =
                message.role ===
                "error";

              return (
                <div
                  key={
                    message.id
                  }
                  className={
                    isUser
                      ? "flex justify-end"
                      : "flex justify-start"
                  }
                >
                  <div
                    className={`
                      max-w-[88%]
                      rounded-2xl
                      px-4
                      py-3
                      text-sm
                      leading-6

                      ${
                        isUser
                          ? `
                            bg-blue-600
                            text-white
                          `
                          : isError
                          ? `
                            border
                            border-red-500/30
                            bg-red-500/10
                            text-red-200
                          `
                          : `
                            border
                            border-slate-800
                            bg-[#0a1725]
                            text-slate-200
                          `
                      }
                    `}
                  >
                    {/* TEXT */}

                    <div
                      className="
                        whitespace-pre-wrap
                        break-words
                      "
                    >
                      {
                        message.text
                      }
                    </div>

                    {/* =========================================
                        REFERENCES
                    ========================================= */}

                    {message.references
                      ?.length >
                      0 && (
                      <div
                        className="
                          mt-3
                          flex
                          flex-wrap
                          gap-2
                          border-t
                          border-slate-700/70
                          pt-3
                        "
                      >
                        {message.references
                          .slice(
                            0,
                            8,
                          )
                          .map(
                            (
                              ref,
                              refIndex,
                            ) => {
                              const refType =
                                String(
                                  ref?.type ||
                                    "record",
                                );

                              const refId =
                                String(
                                  ref?.id ||
                                    refIndex +
                                      1,
                                );

                              const label =
                                `${refType}: ${refId}`;

                              const internalRoute =
                                getInternalRoute(
                                  ref?.href,
                                );

                              const referenceKey =
                                `${refType}-${refId}-${refIndex}`;

                              /*
                               * Route exists:
                               *
                               * use NavigationStackContext
                               * instead of Link.
                               */
                              if (
                                internalRoute
                              ) {
                                return (
                                  <button
                                    key={
                                      referenceKey
                                    }
                                    type="button"
                                    onClick={() =>
                                      openReference(
                                        internalRoute,
                                        ref,
                                      )
                                    }
                                    className="
                                      inline-flex
                                      items-center
                                      gap-1
                                      rounded-full
                                      bg-slate-800
                                      px-2.5
                                      py-1
                                      text-[11px]
                                      text-slate-300
                                      transition
                                      hover:bg-blue-600/30
                                      hover:text-white
                                    "
                                  >
                                    <ExternalLink
                                      size={
                                        11
                                      }
                                    />

                                    {
                                      label
                                    }
                                  </button>
                                );
                              }

                              /*
                               * No navigable route:
                               * display reference only.
                               */

                              return (
                                <span
                                  key={
                                    referenceKey
                                  }
                                  className="
                                    inline-flex
                                    items-center
                                    gap-1
                                    rounded-full
                                    bg-slate-800
                                    px-2.5
                                    py-1
                                    text-[11px]
                                    text-slate-300
                                  "
                                >
                                  <ExternalLink
                                    size={
                                      11
                                    }
                                  />

                                  {
                                    label
                                  }
                                </span>
                              );
                            },
                          )}
                      </div>
                    )}

                    {/* =========================================
                        GENERATED TIME
                    ========================================= */}

                    {message.generatedAt && (
                      <div
                        className="
                          mt-2
                          text-[10px]
                          text-slate-500
                        "
                      >
                        {formatIndianDateTime(message.generatedAt, "")}
                      </div>
                    )}
                  </div>
                </div>
              );
            },
          )}

          {/* =================================================
              LOADING
              ANIMATION ONLY
          ================================================= */}

          {loading && (
            <div
              className="
                flex
                justify-start
              "
            >
              <div
                className="
                  flex
                  h-12
                  min-w-[62px]
                  items-center
                  justify-center
                  rounded-2xl
                  border
                  border-slate-800
                  bg-[#0a1725]
                  px-4
                "
              >
                <LoaderCircle
                  className="
                    animate-spin
                    text-blue-400
                  "
                  size={20}
                />
              </div>
            </div>
          )}

          <div
            ref={
              bottomRef
            }
            className="h-px"
          />
        </div>
      </div>

      {/* ===================================================
          INPUT AREA
      =================================================== */}

      <form
        className="
          shrink-0
          border-t
          border-slate-800
          bg-[#081522]
          p-4
        "
        onSubmit={(
          event,
        ) => {
          event.preventDefault();

          ask();
        }}
      >
        <div
          className="
            mx-auto
            flex
            w-full
            max-w-[1280px]
            items-end
            gap-2
            rounded-2xl
            border
            border-slate-700
            bg-[#050d16]
            p-2
            transition
            focus-within:border-blue-500/60
          "
        >
          <textarea
            value={
              question
            }
            onChange={(
              event,
            ) => {
              setQuestion(
                event.target.value.slice(
                  0,
                  1200,
                ),
              );
            }}
            onKeyDown={(
              event,
            ) => {
              if (
                event.key ===
                  "Enter" &&
                !event.shiftKey
              ) {
                event.preventDefault();

                ask();
              }
            }}
            rows={2}
            maxLength={
              1200
            }
            disabled={
              loading
            }
            placeholder="Ask: What is happening now?"
            className="
              max-h-32
              min-h-[48px]
              flex-1
              resize-none
              bg-transparent
              px-3
              py-2
              text-sm
              text-white
              outline-none
              placeholder:text-slate-600
              disabled:cursor-not-allowed
              disabled:opacity-70
            "
          />

          <button
            type="submit"
            disabled={
              loading ||
              !question.trim()
            }
            aria-label="Send question"
            className="
              flex
              h-11
              w-11
              shrink-0
              items-center
              justify-center
              rounded-xl
              bg-blue-600
              text-white
              transition
              hover:bg-blue-500
              disabled:cursor-not-allowed
              disabled:opacity-40
            "
          >
            {loading ? (
              <LoaderCircle
                className="
                  animate-spin
                "
                size={18}
              />
            ) : (
              <Send
                size={18}
              />
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default IntelligenceAssistant;
