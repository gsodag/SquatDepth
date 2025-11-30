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

    const submitButtonClass = `
        w-full sm:w-auto px-8 py-3 text-lg font-semibold rounded-xl transition duration-300 ease-in-out shadow-lg
        transform hover:scale-105 active:scale-95
        ${loading || !video ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700 text-white focus:outline-none focus:ring-4 focus:ring-red-400/50'}
    `;

    const resultClass = result === "PASS" 
        ? 'bg-gradient-to-br from-green-700 to-emerald-800 text-white border-green-500' 
        : 'bg-gradient-to-br from-red-700 to-rose-800 text-white border-red-500';      

    return (
        <div className="max-w-3xl mx-auto my-10 p-10 bg-zinc-950 text-white shadow-2xl rounded-2xl border-2 border-red-700">
            
            <header className="mb-8 pb-6 border-b border-zinc-700 text-center">
                <h2 className="text-4xl font-extrabold text-white tracking-tight drop-shadow-lg mb-2">
                    Analiza Przysiadu
                </h2>
                <p className="text-zinc-400 text-lg">
                    Wgraj swój film, a sztuczna inteligencja oceni, czy próba kwalifikuje się jako zaliczona.
                </p>
            </header>
            
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
                            <span className="text-3xl mr-4"></span>
                            <p className="font-medium text-xl">Błąd przesyłania: **{error}**</p>
                        </div>
                    )}

                    {result && (
                        <div className={`mt-6 p-10 border-2 rounded-xl text-center shadow-2xl ${resultClass}`}>
                            <h3 className="text-5xl font-extrabold mb-3 animate-pulse">
                                {result === "PASS" ? "PRZYSIAD ZALICZONY!" : "PRZYSIAD NIEZALICZONY"}
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