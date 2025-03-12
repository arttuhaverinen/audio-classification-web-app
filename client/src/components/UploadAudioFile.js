// Component for audio file uploads from user's computer. Other than wav and mp3 causes errors.
const UploadAudioFile = ({
	baseurl,
	setSampleUploaded,
	setEmotions,
	setGender,
	setEmotionsData,
	file,
	setFile,
}) => {
	fetch(`${baseurl}/mem`)
		.then((res) => res.json())
		.then((res) => console.log(res));

	const handleUploadAudioFile = (e) => {
		e.preventDefault();
		setSampleUploaded(true);
		console.log(file);
		let fd = new FormData();
		fd.append("file", file);

		fetch(`${baseurl}/wav`, {
			method: "POST",
			body: fd,
			//headers: { "Content-Type": "multipart/form-data" },
			//mode: "no-cors", // no-cors, *cors, same-origin
		})
			.then((res) => res.json())
			.then((data) => {
				console.log("data values", data.values);
				console.log("data values", data.values[0]);
				console.log("data values", data.values[0][0]);
				setEmotionsData(data.values);
				setEmotions(data.values[0]);
				setGender(data.values[1]);
				setSampleUploaded(false);
			})
			.catch((error) => {
				alert(error);
				setSampleUploaded(false);
				alert("error occured");
			});
	};

	const handlefile = (e) => {
		console.log(e.target.files);
		setFile(e.target.files[0]);
	};

	return (
		<div className="div-upload-audio">
			<div>
				<h2 className="div-audioupload-h1">Upload audio samples</h2>
			</div>
			<form className="form-upload-audio">
				<label for="input-audioupload">Choose a file </label>
				<input
					onChange={(e) => {
						handlefile(e);
					}}
					id="input-audioupload"
					type="file"
					hidden
				/>
				{file ? (
					<button onClick={(e) => handleUploadAudioFile(e)} type="submit">
						Submit file
					</button>
				) : (
					<button disabled type="submit">
						Submit file
					</button>
				)}
			</form>
		</div>
	);
};

export default UploadAudioFile;
