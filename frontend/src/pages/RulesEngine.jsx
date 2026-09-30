import React, { useState } from "react";

const RulesEngine = () => {
  const [activeFilter, setActiveFilter] = useState("ALL");

  const filters = [
    "ALL",
    "CRITICAL",
    "WARNING",
    "MEDIUM"
  ];

  const severityStyles = {
    CRITICAL: {
      border: "border-l-red-500",
      badge: "bg-red-500/20 text-red-400 border border-red-500/40",
    },
    WARNING: {
      border: "border-l-orange-500",
      badge:
        "bg-orange-500/20 text-orange-400 border border-orange-500/40",
    },
    MEDIUM: {
      border:
        "border-l-yellow-500",
      badge:
        "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
    },
    LOW: {
      border: "border-l-blue-500",
      badge: "bg-blue-500/20 text-blue-400 border border-blue-500/40",
    },
  };

  const rules = [
    {
      name: "Sudden Running",
      category: "CRIME",
      severity: "WARNING",
      description:
        "Detects rapid movement patterns indicating running or panic behaviour.",
    },

    {
      name: "Close Interaction",
      category: "CRIME",
      severity: "MEDIUM",
      description:
        "Detects prolonged close proximity between individuals.",
    },

    {
      name: "Loitering",
      category: "CRIME",
      severity: "MEDIUM",
      description:
        "Detects a person remaining within a small area for an extended duration.",
    },

    {
      name: "Repeated Entry Exit",
      category: "CRIME",
      severity: "WARNING",
      description:
        "Detects repeated movement in and out of a monitored zone.",
    },

    {
      name: "Suspicious Behaviour",
      category: "CRIME",
      severity: "WARNING",
      description:
        "Detects unusual movement patterns that deviate from normal behaviour.",
    },

    {
      name: "Suspicious Escalation",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects escalating suspicious activity that may indicate a developing threat.",
    },

    {
      name: "Following Behaviour",
      category: "CRIME",
      severity: "WARNING",
      description:
        "Detects one individual persistently following another person.",
    },

    {
      name: "Chain Snatching Risk",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects movement and interaction patterns associated with chain snatching attempts.",
    },

    {
      name: "Possible Abduction Risk",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects behaviours that may indicate a potential kidnapping or abduction event.",
    },

    {
      name: "Perimeter Breach",
      category: "SECURITY",
      severity: "CRITICAL",
      description:
        "Detects unauthorized entry into restricted or protected areas.",
    },

    {
      name: "Face Covered Walking",
      category: "SECURITY",
      severity: "WARNING",
      description:
        "Detects individuals moving while concealing facial identity.",
    },

    {
      name: "Helmet Walking",
      category: "SECURITY",
      severity: "WARNING",
      description:
        "Detects individuals walking while wearing helmets in monitored areas.",
    },

    {
      name: "Abandoned Object",
      category: "SECURITY",
      severity: "WARNING",
      description:
        "Detects unattended bags, packages, or other suspicious objects.",
    },

    {
      name: "Fighting",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects physical altercations and aggressive interactions.",
    },

    {
      name: "Gun Handling",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects visible firearm possession or weapon handling.",
    },

    {
      name: "Knife Handling",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects visible knife possession or threatening weapon behaviour.",
    },

    {
      name: "Theft / Robbery",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects theft, robbery, or forceful property removal activities.",
    },

    {
      name: "Assault Risk",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects actions indicating a high likelihood of assault or attack.",
    },

    {
      name: "Fall Emergency",
      category: "SECURITY",
      severity: "CRITICAL",
      description:
        "Detects human falls that may require immediate medical assistance.",
    },

    {
      name: "Crime Activity",
      category: "CRIME",
      severity: "CRITICAL",
      description:
        "Detects criminal activity identified by the crime detection model.",
    },

  ];

  const criticalCount = rules.filter(
    (r) => r.severity === "CRITICAL"
  ).length;

  const warningCount = rules.filter(
    (r) => r.severity === "WARNING"
  ).length;

  const mediumCount = rules.filter(
    (r) => r.severity === "MEDIUM"
  ).length;


  const filteredRules =
    activeFilter === "ALL"
      ? rules
      : rules.filter(
        (rule) => rule.severity === activeFilter
      );

  return (
    <div className="w-full bg-[#071827] border border-[#123452] rounded-xl text-white p-6">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
        <div className="flex flex-wrap gap-2">
          {filters.map((item) => {
            const activeClass =
              item === "CRITICAL"
                ? "bg-red-600 text-white"
                : item === "WARNING"
                  ? "bg-orange-500 text-white"
                  : item === "MEDIUM"
                    ? "bg-yellow-500 text-black"
                      : "bg-blue-600 text-white";

            return (
              <button
                key={item}
                onClick={() => setActiveFilter(item)}
                className={`px-5 py-2 rounded-lg text-sm font-bold tracking-wider transition-all ${activeFilter === item
                  ? activeClass
                  : "bg-[#0d2038] border border-[#21456d] text-[#6fa6d6] hover:border-blue-500"
                  }`}
              >
                {item}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">

          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-red-400 text-xs font-semibold uppercase">
              Critical
            </p>
            <h3 className="text-3xl font-bold mt-1">
              {criticalCount}
            </h3>
          </div>

          <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4">
            <p className="text-orange-400 text-xs font-semibold uppercase">
              Warning
            </p>
            <h3 className="text-3xl font-bold mt-1">
              {warningCount}
            </h3>
          </div>

          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
            <p className="text-yellow-400 text-xs font-semibold uppercase">
              Medium
            </p>
            <h3 className="text-3xl font-bold mt-1">
              {mediumCount}
            </h3>
          </div>

          <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
            <p className="text-green-400 text-xs font-semibold uppercase">
              Total Rules
            </p>
            <h3 className="text-3xl font-bold mt-1">
              {rules.length}
            </h3>
          </div>

        </div>
        {filteredRules.map((rule) => (
          <div
            key={rule.name}
            className={`
              bg-[#0d2038]
              border
              border-[#21456d]
              border-l-4
              ${severityStyles[rule.severity].border}
              rounded-xl
              p-5
              hover:border-blue-500
              hover:shadow-lg
              hover:shadow-blue-900/20
              transition-all
            `}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-xl text-white">
                {rule.name}
              </h3>

              <span
                className={`
                  px-3 py-1
                  rounded-full
                  text-xs
                  font-bold
                  tracking-wider
                  ${severityStyles[rule.severity].badge}
                `}
              >
                {rule.severity}
              </span>
            </div>

            <p className="text-[#8dbce5] text-sm leading-relaxed">
              {rule.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RulesEngine;