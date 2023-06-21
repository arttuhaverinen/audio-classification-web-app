// Component for youtube url uploads

const YoutubeUpload = ({
  baseurl,
  setSampleUploaded,
  setEmotions,
  setGender,
  setEmotionsData,
  youtubeLink,
  setYoutubeLink,
}) => {
  const sendYoutubeLink = async () => {
    setSampleUploaded(true);
    fetch(baseurl + "/youtube", {
      method: "POST",
      body: youtubeLink,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setEmotionsData(data.values);
        setEmotions(data.values[0]);
        setGender(data.values[1]);
        setSampleUploaded(false);
      })
      .catch((error) => {
        console.log("error", error.message);
        setSampleUploaded(false);
        alert(error);
      });
  };

  return (
    <div className="div-youtube-upload">
      <h2 className="div-youtubeupload-h1">Upload youtube url</h2>
      <div>
        <input
          id="input-youtubetext"
          type="text"
          placeholder="Enter your youtube link here!"
          onChange={(e) => setYoutubeLink(e.target.value)}
        />
        <button
          onClick={() => {
            sendYoutubeLink();
          }}
          type="submit"
        >
          Submit
        </button>
      </div>
    </div>
  );
};
export default YoutubeUpload;
