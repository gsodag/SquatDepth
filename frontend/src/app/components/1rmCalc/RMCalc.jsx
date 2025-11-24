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
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-zinc-900 to-zinc-800 py-10">
      <div className="bg-zinc-950 rounded-2xl shadow-2xl p-8 w-full max-w-4xl border-2 border-red-700">
        <h1 className="text-4xl font-extrabold text-center mb-4 text-white tracking-tight drop-shadow">1RM & Rep Max Table</h1>
        <form
          className="flex flex-col md:flex-row gap-4 items-center justify-center mb-8"
          onSubmit={e => e.preventDefault()}
        >
          <div className="flex flex-col">
            <label className="text-zinc-200 mb-1">Weight Lifted:</label>
            <input
              type="number"
              value={weight}
              min={0}
              onChange={e => setWeight(e.target.value)}
              className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
              placeholder="0"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-zinc-200 mb-1">Reps Performed:</label>
            <input
              type="number"
              value={reps}
              min={1}
              max={10}
              onChange={e => setReps(e.target.value)}
              className="bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
              placeholder="0"
            />
          </div>
        </form>
        <div className="overflow-x-auto">
          <table className="min-w-full text-center border-separate border-spacing-y-2">
            <thead>
              <tr>
                <th className="text-white text-lg font-semibold">RM</th>
                <th className="text-white text-lg font-semibold">Epley</th>
                <th className="text-white text-lg font-semibold">Brzycki</th>
                <th className="text-white text-lg font-semibold">Lombardi</th>
                <th className="text-white text-lg font-semibold">o'Conner</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => (
                <tr key={row.rm} className="bg-zinc-800 rounded">
                  <td className="text-red-400 font-semibold py-2">{row.rm}</td>
                  <td className="text-zinc-100">{row.epley}</td>
                  <td className="text-zinc-100">{row.brzycki}</td>
                  <td className="text-zinc-100">{row.lombardi}</td>
                  <td className="text-zinc-100">{row.oconner}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-8 text-zinc-400 text-sm text-center">
          <div>
            <b>Estimated 1RM:</b>
          </div>
          <div>
            Epley: <span className="text-white">{oneRM.epley.toFixed(2)} kg</span> |{" "}
            Brzycki: <span className="text-white">{oneRM.brzycki.toFixed(2)} kg</span> |{" "}
            Lombardi: <span className="text-white">{oneRM.lombardi.toFixed(2)} kg</span> |{" "}
            o'Conner: <span className="text-white">{oneRM.oconner.toFixed(2)} kg</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RMCalc;