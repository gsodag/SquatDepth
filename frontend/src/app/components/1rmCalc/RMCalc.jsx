'use client'
import { useState } from "react";

const epley1RM = (w, r) => w * (1 + r / 30);
const brzycki1RM = (w, r) => w / (1.0278 - 0.0278 * r);
const lombardi1RM = (w, r) => w * Math.pow(r, 0.1);
const oconner1RM = (w, r) => w * (1 + 0.025 * r);

const epleyWeight = (oneRM, reps) => oneRM / (1 + reps / 30);
const brzyckiWeight = (oneRM, reps) => oneRM * (1.0278 - 0.0278 * reps);
const lombardiWeight = (oneRM, reps) => oneRM / Math.pow(reps, 0.1);
const oconnerWeight = (oneRM, reps) => oneRM / (1 + 0.025 * reps);

const RMCalc = () => {
  const [weight, setWeight] = useState("");
  const [reps, setReps] = useState("");

  const weightNum = parseFloat(weight) || 0;
  const repsNum = parseInt(reps) || 1;

  const oneRM = {
    epley: epley1RM(weightNum, repsNum),
    brzycki: brzycki1RM(weightNum, repsNum),
    lombardi: lombardi1RM(weightNum, repsNum),
    oconner: oconner1RM(weightNum, repsNum),
  };

  const rows = Array.from({ length: 10 }, (_, i) => {
    const rm = i + 1;
    return {
      rm: `${rm}RM`,
      epley: epleyWeight(oneRM.epley, rm).toFixed(2),
      brzycki: brzyckiWeight(oneRM.brzycki, rm).toFixed(2),
      lombardi: lombardiWeight(oneRM.lombardi, rm).toFixed(2),
      oconner: oconnerWeight(oneRM.oconner, rm).toFixed(2),
    };
  });

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-900 py-10">
      <div className="bg-zinc-950 rounded-2xl shadow-2xl p-8 w-full max-w-4xl border-2 border-red-700">
        <h1 className="text-4xl font-extrabold text-center mb-6 text-white tracking-tight drop-shadow">Kalkulator 1RM & Rep Max</h1>
        <form
          className="flex flex-col md:flex-row gap-6 items-end justify-center mb-10 p-6 bg-zinc-900 rounded-xl shadow-inner border border-zinc-800"
          onSubmit={e => e.preventDefault()}
        >
          <div className="flex flex-col flex-1">
            <label className="text-zinc-300 mb-2 text-lg font-medium">Waga (kg):</label>
            <input
              type="number"
              value={weight}
              min={0}
              onChange={e => setWeight(e.target.value)}
              className="bg-zinc-800 text-white p-3 rounded-xl border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600 shadow-md transition"
              placeholder="0"
            />
          </div>
          <div className="flex flex-col flex-1">
            <label className="text-zinc-300 mb-2 text-lg font-medium">Powtórzenia:</label>
            <input
              type="number"
              value={reps}
              min={1}
              max={10}
              onChange={e => setReps(e.target.value)}
              className="bg-zinc-800 text-white p-3 rounded-xl border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600 shadow-md transition"
              placeholder="0"
            />
          </div>
        </form>

        <div className="p-6 bg-zinc-900 rounded-xl shadow-xl border border-zinc-800">
            <h2 className="text-2xl font-bold mb-4 text-zinc-100 border-b border-zinc-700 pb-2">
                Tabela Rep Max
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-center border-separate border-spacing-y-3">
                <thead>
                  <tr className="bg-zinc-800 rounded-xl shadow-lg">
                    <th className="text-red-400 text-lg font-extrabold py-3 rounded-l-xl">RM</th>
                    <th className="text-zinc-200 text-lg font-semibold">Epley</th>
                    <th className="text-zinc-200 text-lg font-semibold">Brzycki</th>
                    <th className="text-zinc-200 text-lg font-semibold">Lombardi</th>
                    <th className="text-zinc-200 text-lg font-semibold rounded-r-xl">O'Conner</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => (
                    <tr key={row.rm} className="bg-zinc-800/70 hover:bg-zinc-700/80 transition duration-200 shadow-md">
                      <td className="text-red-400 font-extrabold py-3 rounded-l-xl">{row.rm}</td>
                      <td className="text-white">{row.epley}</td>
                      <td className="text-white">{row.brzycki}</td>
                      <td className="text-white">{row.lombardi}</td>
                      <td className="text-white rounded-r-xl">{row.oconner}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        </div>

        <div className="mt-10 p-6 bg-zinc-900 rounded-xl shadow-xl border border-zinc-800 text-center">
            <div className="text-xl font-bold text-zinc-100 mb-4 border-b border-zinc-700 pb-2">
                Szacowane 1RM (kg)
            </div>
            <div className="flex flex-wrap justify-center gap-6 text-lg font-medium">
                <div className="text-zinc-400">
                    Epley: <span className="text-red-400 font-bold">{oneRM.epley.toFixed(2)}</span>
                </div>
                <div className="text-zinc-400">
                    Brzycki: <span className="text-red-400 font-bold">{oneRM.brzycki.toFixed(2)}</span>
                </div>
                <div className="text-zinc-400">
                    Lombardi: <span className="text-red-400 font-bold">{oneRM.lombardi.toFixed(2)}</span>
                </div>
                <div className="text-zinc-400">
                    O'Conner: <span className="text-red-400 font-bold">{oneRM.oconner.toFixed(2)}</span>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default RMCalc;