'use client'

import { useState } from "react";

const barbell = 20;
const plateDefs = [
  { id: "25kg", label: "25 kg", value: 25 },
  { id: "20kg", label: "20 kg", value: 20 },
  { id: "15kg", label: "15 kg", value: 15 },
  { id: "10kg", label: "10 kg", value: 10 },
  { id: "5kg", label: "5 kg", value: 5 },
  { id: "2p5kg", label: "2.5 kg", value: 2.5 },
  { id: "2kg", label: "2 kg", value: 2 },
  { id: "1p5kg", label: "1.5 kg", value: 1.5 },
  { id: "1p25kg", label: "1.25 kg", value: 1.25 },
  { id: "1kg", label: "1 kg", value: 1 },
  { id: "0p5kg", label: "0.5 kg", value: 0.5 },
  { id: "0p25kg", label: "0.25 kg", value: 0.25 },
];

const calculatePlates = (desiredWeight, availablePlates) => {
  if (desiredWeight < barbell) {
    return { error: "Weight must be at least the barbell weight (20 kg)." };
  }

  const tryFind = (target) => {
    let totalWeight = target - barbell;
    let remainingWeight = totalWeight;
    const plateCounts = {};

    for (const plate of plateDefs) {
      const maxPossible = Math.floor(remainingWeight / (plate.value * 2));
      const available = Number(availablePlates[plate.id]) || 0;
      const useCount = Math.min(maxPossible, available);

      plateCounts[plate.id] = useCount;
      remainingWeight -= useCount * plate.value * 2;
    }

    const achievedWeight =
      barbell +
      plateDefs.reduce(
        (sum, plate) => sum + (plateCounts[plate.id] || 0) * plate.value * 2,
        0
      );

    return {
      ...plateCounts,
      achievedWeight,
    };
  };

  let bestBelow = tryFind(desiredWeight);
  let prevBelowWeight = null;
  let belowTries = 0;
  while (
    bestBelow.achievedWeight > desiredWeight &&
    bestBelow.achievedWeight > barbell
  ) {
    if (bestBelow.achievedWeight === prevBelowWeight) break;
    prevBelowWeight = bestBelow.achievedWeight;
    bestBelow = tryFind(bestBelow.achievedWeight - 0.5);
    belowTries++;
    if (belowTries > 1000) break;
  }

  let bestAbove = tryFind(desiredWeight);
  let prevAboveWeight = null;
  let aboveTries = 0;
  while (bestAbove.achievedWeight < desiredWeight) {
    if (bestAbove.achievedWeight === prevAboveWeight) break;
    prevAboveWeight = bestAbove.achievedWeight;
    bestAbove = tryFind(bestAbove.achievedWeight + 0.5);
    aboveTries++;
    if (aboveTries > 1000) break;
  }
  if (bestAbove.achievedWeight <= desiredWeight) bestAbove = null;

  return { bestBelow, bestAbove };
};

const BarbellCalc = () => {
  const [desiredWeight, setDesiredWeight] = useState("");
  const [error, setError] = useState("");
  const [results, setResults] = useState({});
  const [plates, setPlates] = useState(
    plateDefs.reduce((acc, plate) => ({ ...acc, [plate.id]: "" }), {})
  );

  const handleInputChange = (id) => (event) => {
    const value = event.target.value.replace(/[^0-9.,]/g, '').replace(/,/g, '.');
    setPlates((prev) => ({
      ...prev,
      [id]: value,
    }));
  };

  const numericPlates = Object.fromEntries(
    Object.entries(plates).map(([k, v]) => [k, v === "" ? 0 : Number(v)])
  );
  const numericDesiredWeight = desiredWeight === "" ? 0 : Number(desiredWeight);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-zinc-900 to-zinc-800 py-10">
      <div className="bg-zinc-950 rounded-2xl shadow-2xl p-8 w-full max-w-4xl border-2 border-red-700">
        <h1 className="text-4xl font-extrabold text-center mb-4 text-white tracking-tight drop-shadow">Barbell Plate Calculator</h1>
        <p className="text-zinc-300 text-center mb-8">
          Enter your available plates and desired weight. <span className="text-red-400 font-semibold">Red</span> highlights key actions!
        </p>
        <form className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
          <div>
            <h2 className="text-xl font-bold text-zinc-100 mb-3 border-b border-zinc-700 pb-1">Your Plates</h2>
            <div className="flex flex-col gap-3">
              {plateDefs.map((plate) => (
                <div key={plate.id} className="flex items-center gap-3">
                  <label htmlFor={plate.id} className="text-zinc-200 w-20">{plate.label}</label>
                  <input
                    type="number"
                    id={plate.id}
                    value={plates[plate.id]}
                    inputMode="numeric"
                    pattern="[0-9.,]*"
                    onChange={handleInputChange(plate.id)}
                    className="w-20 bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600"
                    autoComplete="off"
                    placeholder="0"
                    min={0}
                  />
                  <span className="text-zinc-400 text-xs">pairs</span>
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-col justify-between">
            <div>
              <h2 className="text-xl font-bold text-zinc-100 mb-3 border-b border-zinc-700 pb-1">Desired Weight</h2>
              <input
                type="number"
                value={desiredWeight}
                onChange={(e) => setDesiredWeight(e.target.value)}
                className="w-full bg-zinc-800 text-white p-2 rounded border border-zinc-700 focus:border-red-500 focus:ring-2 focus:ring-red-600 mb-4"
                placeholder="Weight (kg)"
                min={barbell}
              />
              <div className="text-zinc-400 text-sm mb-2">
                Barbell weight: <span className="font-semibold text-white">{barbell} kg</span>
              </div>
              <button
                className="w-full px-4 py-2 bg-red-600 text-white rounded-lg font-bold hover:bg-red-700 transition shadow-lg"
                onClick={(e) => {
                  e.preventDefault();
                  const result = calculatePlates(numericDesiredWeight, numericPlates);
                  if (result.bestBelow.achievedWeight === barbell && !result.bestAbove) {
                    setResults({});
                    setError("Not enough plates to reach or exceed the desired weight.");
                  } else {
                    setResults(result);
                    setError("");
                  }
                }}
              >
                Calculate Plates
              </button>
            </div>
            <div className="mt-8 flex flex-col items-center">
              <span className="text-zinc-400 text-xs">Plates are counted as <b>pairs</b> for both sides of the bar.</span>
            </div>
          </div>
        </form>
        <div className="border-t border-zinc-700 pt-6">
          <h2 className="text-xl font-bold text-zinc-100 mb-4">Calculated Plates</h2>
          {error ? (
            <div className="text-red-400 mb-2 font-semibold">{error}</div>
          ) : (
            <>
              {results.bestBelow && (
                <div className="mb-6">
                  <div className="text-red-400 mb-1 text-base font-semibold">
                    Closest <b>below or equal</b>: <b>{results.bestBelow.achievedWeight} kg</b>
                  </div>
                  <ul className="list-disc pl-6">
                    {plateDefs
                      .filter((plate) => (results.bestBelow[plate.id] || 0) > 0)
                      .map((plate) => (
                        <li key={plate.id} className="text-zinc-200">
                          {plate.label}: <span className="font-semibold text-red-400">{results.bestBelow[plate.id]}</span> pairs
                        </li>
                      ))}
                  </ul>
                </div>
              )}
              {results.bestAbove && (
                <div className="mb-2">
                  <div className="text-green-400 mb-1 text-base font-semibold">
                    Closest <b>above</b>: <b>{results.bestAbove.achievedWeight} kg</b>
                  </div>
                  <ul className="list-disc pl-6">
                    {plateDefs
                      .filter((plate) => (results.bestAbove[plate.id] || 0) > 0)
                      .map((plate) => (
                        <li key={plate.id} className="text-zinc-200">
                          {plate.label}: <span className="font-semibold text-green-400">{results.bestAbove[plate.id]}</span> pairs
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default BarbellCalc;