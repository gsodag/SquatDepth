'use client';
import React, { useState } from "react";

function UploadForm() {

    const [video, setVideo] = useState(null);
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [model, setModel] = useState('comfort'); // 'comfort' or 'accuracy'

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!video) {
            setError("Proszę wybrać plik video");
            return;
        }

        setLoading(true);
        setError("");
        setResult("");

        try {
            const formData = new FormData();
            formData.append("file", video);
            formData.append("model", model);

            const res = await fetch("http://localhost:8001/upload", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}));
                const errorMessage = errorData.detail || `HTTP error! status: ${res.status}`;
                throw new Error(errorMessage);
            }

            const data = await res.json();

            if (data.prediction) {
                setResult(data.prediction);
            } else if (data.detail) {
                setError(data.detail);
            }

        } catch (err) {
            console.error("Upload error:", err);
            setError(`Błąd podczas przesyłania: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const submitButtonClass = `
        w-full sm:w-auto px-8 py-3 text-lg font-semibold rounded-xl transition duration-300 ease-in-out shadow-lg
        transform hover:scale-105 active:scale-95
        ${loading || !video ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700 text-white focus:outline-none focus:ring-4 focus:ring-red-400/50'}
    `;

    const resultClass = result === "PASS"
        ? 'bg-gradient-to-br from-green-700 to-emerald-800 text-white border-green-500'
        : 'bg-gradient-to-br from-red-700 to-rose-800 text-white border-red-500';

    // Komponent Tooltip z nowym pozycjonowaniem
    const Tooltip = ({ content }) => {
        const [isVisible, setIsVisible] = useState(false);

        return (
            // Pozycjonowanie dymka w prawym górnym rogu z marginesem
            <div
                className="absolute top-0 right-0 p-2"
                onMouseEnter={() => setIsVisible(true)}
                onMouseLeave={() => setIsVisible(false)}
            >
                {/* Ikona 'i' lub '?' */}
                <svg className="h-4 w-4 text-zinc-400 hover:text-red-300 transition duration-150 cursor-pointer" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>

                {/* Dymek - teraz pozycjonowany na lewo od ikony, aby był wewnątrz formularza */}
                {isVisible && (
                    <div className="absolute z-10 w-64 p-3 right-full top-1/2 -translate-y-1/2 mr-2
                                    text-sm text-zinc-100 bg-zinc-800 rounded-lg shadow-2xl
                                    border border-zinc-700 opacity-100 transition duration-300 pointer-events-none">
                        {content}
                        {/* Strzałka dymka - ustawiona na prawo, by wskazywała na ikonę */}
                        <div className="absolute w-0 h-0 border-y-8 border-y-transparent border-l-8 border-l-zinc-800 right-[-8px] top-1/2 -translate-y-1/2"></div>
                    </div>
                )}
            </div>
        );
    };


    const ModelButton = ({ value, label, description, tooltipContent }) => (
        // Dodanie 'relative' jest kluczowe, aby Tooltip (który jest 'absolute') był pozycjonowany względem tego elementu.
        <label
            className={`
                relative flex-1 p-5 rounded-xl cursor-pointer transition duration-300 ease-in-out border-2
                shadow-lg hover:shadow-red-500/50
                ${model === value
                    ? 'bg-red-700 border-red-500 text-white transform scale-[1.02]'
                    : 'bg-zinc-800 border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:border-red-600'
                }
            `}
        >
            {/* Tooltip w prawym górnym rogu */}
            <Tooltip content={tooltipContent} />
            
            <input
                type="radio"
                name="modelSelection"
                value={value}
                checked={model === value}
                onChange={() => setModel(value)}
                className="sr-only"
            />
            <div className="flex flex-col items-start">
                <span className="text-xl font-bold uppercase mb-1 flex items-center">
                    {label}
                    {model === value && (
                        <svg className="h-5 w-5 ml-2 text-white" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path>
                        </svg>
                    )}
                </span>
                <p className={`text-sm ${model === value ? 'text-zinc-200' : 'text-zinc-400'}`}>{description}</p>
            </div>
        </label>
    );


    return (
        <div className="max-w-3xl mx-auto my-10 p-10 bg-zinc-950 text-white shadow-2xl rounded-2xl border-2 border-red-700">

            <header className="mb-8 pb-6 border-b border-zinc-700 text-center">
                <h2 className="text-4xl font-extrabold text-white tracking-tight drop-shadow-lg mb-2">
                    Analiza Przysiadu 🏋️
                </h2>
                <p className="text-zinc-400 text-lg">
                    Wgraj swój film, a sztuczna inteligencja oceni, czy próba kwalifikuje się jako zaliczona.
                </p>
            </header>

            <div className="mb-12 p-8 bg-zinc-900 rounded-2xl shadow-xl border border-zinc-800">
                <h3 className="text-2xl font-bold mb-6 text-zinc-100 border-b border-zinc-700 pb-3">
                    Wybór Modelu Analizy
                </h3>
                <div className="flex flex-col sm:flex-row gap-4">
                    <ModelButton
                        value="comfort"
                        label="Comfort"
                        description="Mniejsze wymagania dotyczące filmu. Mniej precyzyjna analiza."
                        tooltipContent="Model Comfort pozwala na większą elastyczność kąta nagrań (nagrania od przodu oraz 45 stopni do boków), oraz nie narzuca wysokości nagrania, za to charakteryzuje się mniejszą precyzją."
                    />
                    <ModelButton
                        value="accuracy"
                        label="Accuracy"
                        description="Większe wymagania dotyczące filmu. Bardziej precyzyjna analiza."
                        tooltipContent="Model Accuracy wymaga nagrania wykonanego dokładnie od boku, oraz preferowanej wysokości nagrania na poziomie kolana. W zamian oferuje najwyższą precyzję analizy spośród dostępnych modeli."
                    />
                </div>
                <p className="mt-4 text-sm text-zinc-400">Aktualnie wybrany model: <strong className="text-red-400">{model.toUpperCase()}</strong></p>
            </div>

            <form onSubmit={handleSubmit} className="mb-12 p-8 bg-zinc-900 rounded-2xl shadow-xl border border-zinc-800">
                <h3 className="text-2xl font-bold mb-6 text-zinc-100 border-b border-zinc-700 pb-3">
                    Krok 1: Dodaj Film
                </h3>

                <div className="flex flex-col md:flex-row items-stretch md:items-center gap-8">

                    <div className="flex-grow">
                        <label className="block text-base font-medium text-zinc-300 mb-3">
                            Wybierz plik wideo (Preferowana maksymalna długość filmu: 5 sekund)
                        </label>
                        <input
                            type="file"
                            accept="video/*"
                            onChange={(e) => setVideo(e.target.files[0])}
                            disabled={loading}
                            className="block w-full text-base text-zinc-300
                                file:mr-4 file:py-3 file:px-6
                                file:rounded-xl file:border-0
                                file:text-base file:font-semibold
                                file:bg-zinc-700 file:text-white
                                hover:file:bg-zinc-600 disabled:file:bg-zinc-800 disabled:file:text-zinc-500
                                bg-zinc-800/50 rounded-xl p-3 cursor-pointer transition duration-200 focus:outline-none focus:ring-2 focus:ring-red-500
                            "
                        />
                        {video && (
                            <p className="mt-4 text-sm text-zinc-400 truncate">Wybrano: **{video.name}**</p>
                        )}
                    </div>

                    <button
                        type="submit"
                        disabled={loading || !video}
                        className={submitButtonClass}
                    >
                        {loading ? (
                            <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Analizuję...
                            </>
                        ) : "Krok 2: Analizuj Film"}
                    </button>
                </div>
            </form>

            <div className="p-8 bg-zinc-900 rounded-2xl shadow-xl border border-zinc-800">
                <h3 className="text-2xl font-bold mb-6 text-zinc-100 border-b border-zinc-700 pb-3">
                    Status i Rezultat
                </h3>

                <p className="text-base mb-8 border-l-4 border-yellow-500 pl-4 bg-zinc-800 text-yellow-300 rounded-r-md py-3 shadow-md">
                    <span className="font-semibold">Czekam na dane:</span> Po wysłaniu filmu tutaj pojawi się rezultat analizy.
                </p>

                <div className="space-y-6">
                    {loading && (
                        <div className="p-6 text-center text-zinc-300 bg-zinc-700 rounded-lg shadow-md border border-zinc-600">
                            <p className="font-medium text-xl">Przetwarzanie filmu... Proszę czekać.</p>
                        </div>
                    )}

                    {error && (
                        <div className="p-6 bg-red-800 text-white border border-red-600 rounded-lg flex items-center shadow-lg">
                            <svg className="h-6 w-6 mr-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <p className="font-medium text-xl">Błąd przesyłania: **{error}**</p>
                        </div>
                    )}

                    {result && (
                        <div className={`mt-6 p-10 border-2 rounded-xl text-center shadow-2xl ${resultClass}`}>
                            <h3 className="text-5xl font-extrabold mb-3 animate-pulse">
                                {result === "PASS" ? "PRZYSIAD ZALICZONY! 🎉" : "PRZYSIAD NIEZALICZONY ❌"}
                            </h3>
                            <p className="text-2xl font-medium">Oficjalny wynik analizy AI: <strong className="uppercase">{result}</strong></p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default UploadForm;