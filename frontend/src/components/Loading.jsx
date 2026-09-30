import React from "react";

const Loading = () => {
  const dots = Array.from({ length: 8 });

  return (
    <div className="flex items-center justify-center min-h-[50vh] bg-transparent">
      <div className="relative w-24 h-24">
        <div className="absolute inset-0 border-4 border-blue-900 rounded-full"></div>

        <div className="absolute inset-0 border-4 border-transparent border-t-cyan-400 rounded-full animate-spin"></div>

        <div className="absolute inset-2 border-4 border-transparent border-t-blue-500 rounded-full animate-spin [animation-direction:reverse] [animation-duration:1.5s]"></div>

        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};

export default Loading;