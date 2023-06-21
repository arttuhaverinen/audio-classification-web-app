const SampleUploaded = ({ sampleUploaded }) => {
  return (
    <div
      className={
        sampleUploaded ? "div-uploaded-sample" : "div-uploaded-sample-hidden"
      }
    >
      {sampleUploaded ? (
        <h1>Analyzing your sample, please wait...</h1>
      ) : (
        <h1>placeholder</h1>
      )}
    </div>
  );
};

export default SampleUploaded;
