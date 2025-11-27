'use client';
import React, {useState} from "react";

function UploadForm(){

    const [video, setVideo] = useState(null);
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

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

    // Klasy dla przycisku wysyłania 
    const submitButtonClass = `
        w-full sm:w-auto px-8 py-3 text-lg font-semibold rounded-lg transition duration-300 ease-in-out shadow-lg
        transform hover:scale-105 active:scale-95
        ${loading || !video ? 'bg-zinc-600 text-zinc-400 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700 text-white focus:outline-none focus:ring-4 focus:ring-red-300'}
    `;

    // Klasy dla wyników
    const resultClass = result === "PASS" 
        ? 'bg-green-700 text-green-100 border-green-500' 
        : 'bg-red-700 text-red-100 border-red-500';      

    return (
        // Główny kontener: dopasowany do stylów kalkulatorów
        <div className="max-w-3xl mx-auto p-10 bg-zinc-950 text-white shadow-2xl rounded-2xl border-2 border-red-700">
            
            {/* Nagłówek Sekcji */}
            <header className="mb-8 pb-4 border-b border-zinc-700">
                <h2 className="text-4xl font-extrabold text-white tracking-tight drop-shadow text-center">
                    Analiza Przysiadu 🏋️
                </h2>
                <p className="text-zinc-400 mt-2 text-center">
                    Wgraj swój film, a sztuczna inteligencja oceni, czy próba kwalifikuje się jako zaliczona.
                </p>
            </header>
            
            {/* 1. SEKCJA WEJŚCIOWA (INPUT) */}
            <form onSubmit={handleSubmit} className="mb-10 p-6 bg-zinc-900 rounded-xl shadow-inner border border-zinc-800">
                <h3 className="text-xl font-bold mb-5 text-zinc-100 border-b border-zinc-700 pb-2">
                    Krok 1: Dodaj Film
                </h3>

                <div className="flex flex-col md:flex-row items-stretch md:items-center gap-6">
                    
                    {/* Pole wyboru pliku */}
                    <div className="flex-grow">
                        <label className="block text-sm font-medium text-zinc-300 mb-2">
                            Wybierz plik wideo (Max. 5 sekund)
                        </label>
                        <input 
                            type="file"
                            accept="video/*" 
                            onChange={(e) => setVideo(e.target.files[0])}
                            disabled={loading}
                            className="block w-full text-sm text-zinc-300 
                                file:mr-4 file:py-3 file:px-6
                                file:rounded-xl file:border-0
                                file:text-base file:font-semibold
                                file:bg-zinc-700 file:text-white
                                hover:file:bg-zinc-600 disabled:file:bg-zinc-800 disabled:file:text-zinc-500
                                bg-zinc-800 rounded-xl p-3 cursor-pointer transition duration-200
                            "
                        />
                        {video && (
                            <p className="mt-3 text-sm text-zinc-400 truncate">Wybrano: **{video.name}**</p>
                        )}
                    </div>
                    
                    {/* Przycisk wysyłania */}
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
                        ) : "Krok 2: Analizuj"}
                    </button>
                </div>
            </form>
            
            {/* 2. SEKCJA STATUSU I WYNIKU (OUTPUT) */}
            <div className="p-6 bg-zinc-900 rounded-xl shadow-inner border border-zinc-800">
                <h3 className="text-xl font-bold mb-5 text-zinc-100 border-b border-zinc-700 pb-2">
                    Status i Rezultat
                </h3>

                {/* Wyróżniona informacja */}
                <p className="text-sm mb-6 border-l-4 border-yellow-500 pl-4 bg-zinc-800 text-yellow-300 rounded-r-md py-3 shadow-md">
                    <span className="font-semibold">Czekam na dane:</span> Po wysłaniu filmu tutaj pojawi się rezultat analizy.
                </p>
                
                <div className="space-y-4">
                    {/* Komunikat ładowania */}
                    {loading && (
                        <div className="p-5 text-center text-zinc-400 bg-zinc-700 rounded-lg shadow-md border border-zinc-600">
                            <p className="font-medium text-lg">⏳ Przetwarzanie filmu... Proszę czekać.</p>
                        </div>
                    )}

                    {/* Komunikat błędu */}
                    {error && (
                        <div className="p-5 bg-red-800 text-red-100 border border-red-600 rounded-lg flex items-center shadow-lg">
                            <span className="text-2xl mr-4">❌</span>
                            <p className="font-medium text-lg">Błąd przesyłania: **{error}**</p>
                        </div>
                    )}

                    {/* Wynik analizy */}
                    {result && (
                        <div className={`mt-6 p-8 border-2 rounded-xl text-center shadow-2xl ${resultClass}`}>
                            <h3 className="text-4xl font-extrabold mb-3">
                                {result === "PASS" ? "✅ RUCH ZALICZONY!" : "❌ RUCH NIEZALICZONY"}
                            </h3>
                            <p className="text-xl font-medium">Oficjalny wynik analizy AI: <strong className="uppercase">{result}</strong></p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default UploadForm;