'use client';
import React, { useState } from "react";

function UploadForm() {
    const [video, setVideo] = useState(null);
    const [model, setModel] = useState("accuracy");
    const [result, setResult] = useState("");
    const [videoUrl, setVideoUrl] = useState("");
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
        setVideoUrl("");

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
                if (data.video_url) {
                    setVideoUrl(data.video_url);
                }
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
        ${loading || !video ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700 text-white'}
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
                    Wgraj swój film, a sztuczna inteligencja oceni twoją technikę.
                </p>
            </header>

            <div className="mb-8 p-6 bg-zinc-900 rounded-xl border border-zinc-800">
                <h3 className="text-xl font-bold mb-4 text-zinc-100">Wybierz tryb analizy:</h3>
                <div className="flex flex-col sm:flex-row gap-4">
                    <label className={`flex-1 p-4 rounded-lg cursor-pointer border-2 transition-all ${model === 'comfort' ? 'border-red-500 bg-zinc-800' : 'border-zinc-700 hover:border-zinc-500'}`}>
                        <div className="flex items-center mb-2">
                            <input
                                type="radio"
                                name="model"
                                value="comfort"
                                checked={model === 'comfort'}
                                onChange={(e) => setModel(e.target.value)}
                                className="w-5 h-5 text-red-600 focus:ring-red-500"
                            />
                            <span className="ml-3 font-bold text-lg">Comfort Mode</span>
                        </div>
                        <p className="text-sm text-zinc-400 ml-8">
                            Mniejsze wymagania co do nagrania. Działa przy lekkim kącie kamery.
                        </p>
                    </label>

                    <label className={`flex-1 p-4 rounded-lg cursor-pointer border-2 transition-all ${model === 'accuracy' ? 'border-red-500 bg-zinc-800' : 'border-zinc-700 hover:border-zinc-500'}`}>
                        <div className="flex items-center mb-2">
                            <input
                                type="radio"
                                name="model"
                                value="accuracy"
                                checked={model === 'accuracy'}
                                onChange={(e) => setModel(e.target.value)}
                                className="w-5 h-5 text-red-600 focus:ring-red-500"
                            />
                            <span className="ml-3 font-bold text-lg">Accuracy Mode</span>
                        </div>
                        <p className="text-sm text-zinc-400 ml-8">
                            Wymaga nagrania idealnie z boku. Wyższa precyzja analizy głębokości.
                        </p>
                    </label>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="mb-12 p-8 bg-zinc-900 rounded-2xl shadow-xl border border-zinc-800">
                <div className="flex flex-col md:flex-row items-stretch md:items-center gap-8">
                    <div className="flex-grow">
                        <label className="block text-base font-medium text-zinc-300 mb-3">
                            Wybierz plik wideo (Max 5-8 sekund)
                        </label>
                        <input
                            type="file"
                            accept="video/*"
                            onChange={(e) => setVideo(e.target.files[0])}
                            disabled={loading}
                            className="block w-full text-sm text-zinc-400
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-full file:border-0
                                file:text-sm file:font-semibold
                                file:bg-zinc-700 file:text-white
                                hover:file:bg-zinc-600
                                cursor-pointer"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading || !video}
                        className={submitButtonClass}
                    >
                        {loading ? "Analizuję..." : "Analizuj"}
                    </button>
                </div>
            </form>

            {(result || error || loading) && (
                <div className="p-8 bg-zinc-900 rounded-2xl shadow-xl border border-zinc-800">
                    <h3 className="text-2xl font-bold mb-6 text-zinc-100 border-b border-zinc-700 pb-3">
                        Status i Rezultat
                    </h3>

                    {loading && (
                        <div className="text-center text-zinc-400 py-4">
                            <p className="animate-pulse">Przetwarzanie filmu... Proszę czekać.</p>
                        </div>
                    )}

                    {error && (
                        <div className="p-4 bg-red-900/30 text-red-200 border border-red-800 rounded-lg">
                            ❌ {error}
                        </div>
                    )}

                    {result && (
                        <div className={`p-8 border-2 rounded-xl text-center shadow-2xl ${resultClass}`}>
                            <h3 className="text-5xl font-extrabold mb-2">
                                {result === "PASS" ? "ZALICZONE! 🎉" : "NIEZALICZONE"}
                            </h3>
                            <p className="text-xl opacity-90">Model: {model.toUpperCase()}</p>
                        </div>
                    )}

                    {videoUrl && (
                        <div className="mt-8">
                            <h4 className="text-lg font-bold mb-2 text-zinc-300">Wizualizacja analizy:</h4>
                            <div className="rounded-xl overflow-hidden shadow-lg border border-zinc-700">
                                <video controls className="w-full bg-black" src={videoUrl}>
                                    Twoja przeglądarka nie obsługuje wideo.
                                </video>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default UploadForm;