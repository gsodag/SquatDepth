'use client'
import { useState } from "react";

const DotsCalc = () => {
  const [bw, setBw] = useState("");
  const [squat, setSquat] = useState("");
  const [bench, setBench] = useState("");
  const [deadlift, setDeadlift] = useState("");
  const [gender, setGender] = useState("male");

  const bwNum = parseFloat(bw) || 0;
  const squatNum = parseFloat(squat) || 0;
  const benchNum = parseFloat(bench) || 0;
  const deadliftNum = parseFloat(deadlift) || 0;
  const total = squatNum + benchNum + deadliftNum;

  let dots = 0;
  if (bwNum > 0) {
    let A, B, C, D, E;
    if (gender === "male") {
      A = -0.000001093;
      B = 0.0007391293;
      C = -0.1918759221;
      D = 24.0900756;
      E = -307.75076;
    } else {
      A = -0.0000010706;
      B = 0.0005158568;
      C = -0.1126655495;
      D = 13.6175032;
      E = -57.96288;
    }
    const denominator =
      A * Math.pow(bwNum, 4) +
      B * Math.pow(bwNum, 3) +
      C * Math.pow(bwNum, 2) +
      D * bwNum +
      E;
    dots = denominator !== 0 ? (total * 500) / denominator : 0;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-zinc-900 to-zinc-800 py-10">
      <div className="bg-zinc-950 rounded-2xl shadow-2xl p-8 w-full max-w-2xl border-2 border-red-700">
        <h1 className="text-4xl font-extrabold text-center mb-4 text-white tracking-tight drop-shadow">DOTS Calculator</h1>
        <form className="flex flex-col gap-6 mb-8">
          <div className="flex gap-4">
            <label className="text-zinc-200 font-semibold">Gender:</label>
            <select
              value={gender}
              onChange={e => setGender(e.target.value)}
              className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex flex-col flex-1">
              <label className="text-zinc-200">Body Weight (kg):</label>
              <input
                type="number"
                value={bw}
                onChange={e => setBw(e.target.value)}
                className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
                placeholder="0"
                min={0}
              />
            </div>
            <div className="flex flex-col flex-1">
              <label className="text-zinc-200">Squat (kg):</label>
              <input
                type="number"
                value={squat}
                onChange={e => setSquat(e.target.value)}
                className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
                placeholder="0"
                min={0}
              />
            </div>
          </div>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex flex-col flex-1">
              <label className="text-zinc-200">Bench (kg):</label>
              <input
                type="number"
                value={bench}
                onChange={e => setBench(e.target.value)}
                className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
                placeholder="0"
                min={0}
              />
            </div>
            <div className="flex flex-col flex-1">
              <label className="text-zinc-200">Deadlift (kg):</label>
              <input
                type="number"
                value={deadlift}
                onChange={e => setDeadlift(e.target.value)}
                className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
                placeholder="0" 
                min={0}
              />
            </div>
          </div>
        </form>
        <div className="border-t border-zinc-700 pt-6 text-center">
          <div className="text-xl font-bold text-zinc-100 mb-2">Your DOTS Score</div>
          <div className="text-3xl font-extrabold text-red-500 drop-shadow">{dots ? dots.toFixed(2) : "--"}</div>
          <div className="text-zinc-400 mt-2">Total: <span className="text-white">{total}</span> kg</div>
        </div>
      </div>
    </div>
  );
};

export default DotsCalc;