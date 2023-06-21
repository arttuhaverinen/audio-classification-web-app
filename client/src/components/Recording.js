// Component for uploading browser recorded audio. Tested only with chrome browser.
import { useReactMediaRecorder } from "react-media-recorder";

const Recording = ({
  status,
  startRecording,
  stopRecording,
  mediaBlobUrl,
  baseurl,
  setSampleUploaded,
  setEmotions,
  setGender,
  setEmotionsData,
}) => {
  const sendWav = async () => {
    const mediaBlob = await fetch(mediaBlobUrl).then((response) =>
      response.blob()
    );
    const myFile = new File([mediaBlob], "file.wav", {
      type: useReactMediaRecorder.mimeType,
    });
    let fd = new FormData();
    fd.append("audiofile", mediaBlob);
    fd.append("file", "test");

    console.log(fd.get("audiofile"));
    console.log(fd.get("file"));
    setSampleUploaded(true);

    fetch(baseurl + "/blob", {
      method: "POST",
      body: fd,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("data values: ", data.values);
        console.log(data.values);
        setEmotionsData(data.values);
        setEmotions(data.values[0]);
        setGender(data.values[1]);
        setSampleUploaded(false);
      })
      .catch((error) => {
        setSampleUploaded(false);
        alert(error);
      });
  };

  return (
    <div className="div-recording">
      <div>
        <h2 className="div-recording-h1">Record your voice</h2>
      </div>
      {status == "recording" ? (
        <h3 id="h3-recording-red">Recording status: {status}</h3>
      ) : (
        <h3>Recording status: {status}</h3>
      )}
      <div className="div-recording-buttons">
        <button onClick={startRecording}>Start Recording</button>
        <button onClick={stopRecording}>Stop Recording</button>
        {mediaBlobUrl == undefined ? (
          <button disabled type="submit">
            Submit
          </button>
        ) : (
          <>
            <button onClick={() => sendWav()} type="submit">
              Submit
            </button>
          </>
        )}
        <audio src={mediaBlobUrl} controls />
      </div>
    </div>
  );
};

export default Recording;
