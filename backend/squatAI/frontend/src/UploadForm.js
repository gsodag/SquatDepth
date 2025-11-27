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
                throw new Error(`HTTP error! status: ${res.status}`);
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

return (
        <div>
            <h3>Wgraj swój filmik z przysiadem, a dowiesz się, czy ta próba będzie zaliczona na zawodach.</h3>
            <p>Najlepiej, aby twój filmik trwał do 5 sekund.</p>
            
            <div className="buttons">
                <div>
                    <input 
                        type="file"
                        accept="video/*" 
                        onChange={(e) => setVideo(e.target.files[0])}
                        disabled={loading}
                        style={{ marginBottom: "10px" }}
                    />
                </div>
                
                <button 
                    onClick={handleSubmit}
                    disabled={loading || !video}
                    style={{
                        padding: "10px 20px",
                        backgroundColor: loading ? "#ccc" : "#007bff",
                        color: "white",
                        border: "none",
                        borderRadius: "4px",
                        cursor: loading ? "not-allowed" : "pointer"
                    }}
                >
                    {loading ? "Analizuję..." : "Dodaj film"}
                </button>
            </div>

            {loading && (
                <div style={{ textAlign: "center", color: "#666" }}>
                    <p>⏳ Przetwarzanie filmu... To może potrwać chwilę.</p>
                </div>
            )}

            {error && (
                <div style={{ 
                    backgroundColor: "#f8d7da", 
                    color: "#721c24", 
                    padding: "10px", 
                    borderRadius: "4px",
                    marginBottom: "15px"
                }}>
                    ❌ {error}
                </div>
            )}

            {result && (
                <div style={{ 
                    backgroundColor: result === "PASS" ? "#d4edda" : "#f8d7da",
                    color: result === "PASS" ? "#155724" : "#721c24",
                    padding: "15px", 
                    borderRadius: "4px",
                    textAlign: "center"
                }}>
                    <h3>
                        {result === "PASS" ? "✅ ZALICZONE!" : "❌ NIEZALICZONE"}
                    </h3>
                    <p>Wynik analizy: <strong>{result}</strong></p>
                </div>
            )}
        </div>
    );
}

export default UploadForm;