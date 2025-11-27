'use client'

import { useState } from "react";

const barbell = 20;
const plateDefs = [
  { id: "25kg", label: "25 kg", value: 25, color: 'bg-red-600', thickness: 'w-6', heightClass: 'h-32' }, 
  { id: "20kg", label: "20 kg", value: 20, color: 'bg-blue-600', thickness: 'w-5', heightClass: 'h-32' },
  { id: "15kg", label: "15 kg", value: 15, color: 'bg-yellow-300', thickness: 'w-4', heightClass: 'h-30' },
  { id: "10kg", label: "10 kg", value: 10, color: 'bg-green-600', thickness: 'w-3', heightClass: 'h-28' },
  { id: "5kg", label: "5 kg", value: 5, color: 'bg-white text-zinc-900', thickness: 'w-3', heightClass: 'h-24' },
  { id: "2p5kg", label: "2.5 kg", value: 2.5, color: 'bg-black', thickness: 'w-2', heightClass: 'h-20' },
  { id: "2kg", label: "2 kg", value: 2, color: 'bg-blue-600', thickness: 'w-2', heightClass: 'h-20' },
  { id: "1p5kg", label: "1.5 kg", value: 1.5, color: 'bg-yellow-300', thickness: 'w-1.5', heightClass: 'h-16' },
  { id: "1p25kg", label: "1.25 kg", value: 1.25, color: 'bg-white', thickness: 'w-1.5', heightClass: 'h-14' },
  { id: "1kg", label: "1 kg", value: 1, color: 'bg-green-600', thickness: 'w-1.5', heightClass: 'h-12' },
  { id: "0p5kg", label: "0.5 kg", value: 0.5, color: 'bg-white', thickness: 'w-1', heightClass: 'h-10' },
  { id: "0p25kg", label: "0.25 kg", value: 0.25, color: 'bg-white', thickness: 'w-1', heightClass: 'h-8' },
];

const calculatePlates = (desiredWeight, availablePlates) => {
  if (desiredWeight < barbell) {
    return { error: "Waga musi wynosić co najmniej wagę sztangi (20 kg)." };
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


const BarbellVisualization = ({ plateResults, plateDefs }) => {
  if (!plateResults) return null;

  const platesToRender = plateDefs
    .flatMap(plateDef => {
      const count = plateResults[plateDef.id] || 0;
      return Array(count).fill({ ...plateDef });
    });
    
  const sleeveWidth = 'w-2';

  return (
    <div className="flex justify-center items-center py-8 px-4 bg-zinc-900 rounded-xl shadow-inner overflow-x-auto min-h-[180px]"> 
      
      <div className="flex flex-row-reverse items-center justify-end">

        <div className={`bg-zinc-700 h-4 ${sleeveWidth} shadow-md flex-shrink-0 rounded-r-md`}></div>

        {platesToRender.map((plate, index) => (
          <div
          key={`left-${index}-${plate.id}`}
          className={`${plate.heightClass} ${plate.thickness} ${plate.color} shadow-lg flex-shrink-0 transition-all duration-500 relative`}
          style={{ 
            zIndex: 100 - index, 
            marginRight: '-0.1rem',
              transform: `translateX(${index * 0.5}px)`
            }}
            >
            {plate.value >= 10 && (
              <span className={`absolute inset-0 flex items-center justify-center text-xs font-bold ${plate.color.includes('white') ? 'text-zinc-900' : 'text-white'}`}>
                    {plate.label.replace(' kg', '')}
                </span>
            )}
          </div>
        ))}

        <div className="bg-zinc-800 h-4 w-4 rounded-l-sm shadow-xl flex-shrink-0 border-r border-zinc-700"></div>
      </div>
      
      <div className={`bg-zinc-600 h-1 w-64 shadow-inner flex-shrink-0`}></div>

      <div className="flex flex-row items-center justify-start">
        
        <div className={`bg-zinc-700 h-4 ${sleeveWidth} shadow-md flex-shrink-0 rounded-l-md`}></div>
        
        {platesToRender.map((plate, index) => (
          <div
            key={`right-${index}-${plate.id}`}
            className={`${plate.heightClass} ${plate.thickness} ${plate.color} shadow-lg flex-shrink-0 transition-all duration-500 relative`}
            style={{ 
              zIndex: 100 - index, 
              marginLeft: '-0.1rem',
              transform: `translateX(${index * -0.5}px)`
            }}
          >
            {plate.value >= 10 && (
                <span className={`absolute inset-0 flex items-center justify-center text-xs font-bold ${plate.color.includes('white') ? 'text-zinc-900' : 'text-white'}`}>
                    {plate.label.replace(' kg', '')}
                </span>
            )}
          </div>
        ))}
         
         <div className="bg-zinc-800 h-4 w-4 rounded-r-sm shadow-xl flex-shrink-0 border-l border-zinc-700"></div>
      </div>
    </div>
  );
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
    <div className="min-h-screen flex items-center justify-center bg-zinc-900 py-10">
      <div className="bg-zinc-950 rounded-2xl shadow-2xl p-8 w-full max-w-5xl border-2 border-red-700">
        <h1 className="text-4xl font-extrabold text-center mb-4 text-white tracking-tight drop-shadow">Kalkulator Obciążenia Sztangi</h1>
        <p className="text-zinc-400 text-center mb-8 text-lg">
          Wprowadź dostępne obciążenie i pożądaną wagę, aby obliczyć, ile talerzy założyć.
        </p>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10 p-6 bg-zinc-900 rounded-xl shadow-xl border border-zinc-800">
          
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-zinc-100 mb-3 border-b border-zinc-700 pb-2">1. Dostępne Talerze (pary)</h2>
            <div className="flex flex-col gap-3 max-h-96 overflow-y-auto pr-2">
              {plateDefs.map((plate) => (
                <div key={plate.id} className="flex items-center gap-4 bg-zinc-800 p-3 rounded-lg border border-zinc-700">
                  <span className={`w-4 h-4 rounded-full ${plate.color} shadow-lg flex-shrink-0`}></span>
                  <label htmlFor={plate.id} className="text-zinc-200 w-24 font-semibold">{plate.label}</label>
                  <input
                    type="number"
                    id={plate.id}
                    value={plates[plate.id]}
                    inputMode="numeric"
                    pattern="[0-9.,]*"
                    onChange={handleInputChange(plate.id)}
                    className="w-24 bg-zinc-700 text-white p-2 rounded-xl border border-zinc-600 focus:border-red-500 focus:ring-2 focus:ring-red-600 transition"
                    autoComplete="off"
                    placeholder="0"
                    min={0}
                  />
                  <span className="text-zinc-400 text-sm">pary</span>
                </div>
              ))}
            </div>
            <div className="text-zinc-500 text-xs pt-2 border-t border-zinc-800">
              *Talerze liczone są w parach (na obie strony sztangi).
            </div>
          </div>
          
          <div className="flex flex-col justify-between space-y-8">
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-zinc-100 mb-3 border-b border-zinc-700 pb-2">2. Oblicz Wagę Docelową</h2>
              <label className="block text-zinc-300 font-medium">Pożądana Waga (kg):</label>
              <input
                type="number"
                value={desiredWeight}
                onChange={(e) => setDesiredWeight(e.target.value)}
                className="w-full bg-zinc-700 text-white p-3 rounded-xl border border-zinc-600 focus:border-red-500 focus:ring-2 focus:ring-red-600 shadow-md mb-4 transition"
                placeholder="Waga (kg)"
                min={barbell}
              />
              <div className="text-zinc-400 text-sm mb-6 p-3 bg-zinc-800 rounded-lg">
                Waga samej sztangi: <span className="font-semibold text-white">{barbell} kg</span>
              </div>
              
              <button
                className="w-full px-6 py-3 bg-red-600 text-white rounded-xl text-lg font-bold hover:bg-red-700 transition shadow-lg transform hover:scale-[1.02] disabled:bg-zinc-700 disabled:text-zinc-400"
                onClick={(e) => {
                  e.preventDefault();
                  const result = calculatePlates(numericDesiredWeight, numericPlates);
                  if (result.error) {
                    setResults({});
                    setError(result.error);
                  } else if (result.bestBelow.achievedWeight === barbell && !result.bestAbove) {
                    setResults({});
                    setError("Brak wystarczającej liczby talerzy, aby osiągnąć lub przekroczyć pożądaną wagę.");
                  } else {
                    setResults(result);
                    setError("");
                  }
                }}
                disabled={!numericDesiredWeight || numericDesiredWeight < barbell}
              >
                Oblicz Obciążenie
              </button>
            </div>
          </div>
        </form>

        {results.bestBelow && (
          <div className="mb-8 border-b border-zinc-700 pb-6">
             <h2 className="text-2xl font-bold text-zinc-100 mb-4 border-b border-zinc-700 pb-2">Wizualizacja</h2>
             <BarbellVisualization 
                 plateResults={results.bestBelow} 
                 plateDefs={plateDefs} 
             />
             {results.bestAbove && (
                <p className="text-sm text-zinc-400 text-center mt-3">
                    Powyżej pokazano wizualizację najbliższej wagi *równiej lub mniejszej* ({results.bestBelow.achievedWeight} kg).
                </p>
             )}
          </div>
        )}

        <div className="pt-8 p-6 bg-zinc-900 rounded-xl shadow-xl border border-zinc-800">
          <h2 className="text-2xl font-bold text-zinc-100 mb-4">Szczegóły Obliczeń</h2>
          {error ? (
            <div className="p-4 bg-red-800 text-white border border-red-600 rounded-lg font-semibold">{error}</div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {results.bestBelow && (
                <div className="p-4 bg-zinc-800 rounded-lg border border-zinc-700 shadow-md">
                  <div className="text-red-400 mb-2 text-xl font-bold border-b border-zinc-700 pb-2">
                    Najbliższa waga <b>(&le;)</b>: <span className="text-white">{results.bestBelow.achievedWeight} kg</span>
                  </div>
                  <ul className="pl-4 space-y-1">
                    {plateDefs
                      .filter((plate) => (results.bestBelow[plate.id] || 0) > 0)
                      .map((plate) => (
                        <li key={plate.id} className="text-zinc-300 flex items-center gap-2">
                          <span className={`w-3 h-3 rounded-full ${plate.color}`}></span>
                          {plate.label}: <span className="font-bold text-red-400">{results.bestBelow[plate.id]}</span> pary
                        </li>
                      ))}
                  </ul>
                </div>
              )}
              {results.bestAbove && (
                <div className="p-4 bg-zinc-800 rounded-lg border border-zinc-700 shadow-md">
                  <div className="text-green-400 mb-2 text-xl font-bold border-b border-zinc-700 pb-2">
                    Najbliższa waga <b>(&gt;)</b>: <span className="text-white">{results.bestAbove.achievedWeight} kg</span>
                  </div>
                  <ul className="pl-4 space-y-1">
                    {plateDefs
                      .filter((plate) => (results.bestAbove[plate.id] || 0) > 0)
                      .map((plate) => (
                        <li key={plate.id} className="text-zinc-300 flex items-center gap-2">
                          <span className={`w-3 h-3 rounded-full ${plate.color}`}></span>
                          {plate.label}: <span className="font-bold text-green-400">{results.bestAbove[plate.id]}</span> pary
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BarbellCalc;