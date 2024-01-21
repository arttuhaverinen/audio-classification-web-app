import "./App.css";
import React, { useState, useEffect } from "react";
import { useReactMediaRecorder } from "react-media-recorder";
import ChartCard from "./components/ChartCard";
import { Chart as ChartJS } from "chart.js/auto";
import { CSVLink, CSVDownload } from "react-csv";
import { useTransition } from "react";
import Settings from "./components/Settings";
import Recording from "./components/Recording";
import SampleUploaded from "./components/SampleUploaded";
import handleCsvCreation from "./utils/handleCsvCreation";
import UploadAudioFile from "./components/UploadAudioFile";
import YoutubeUpload from "./components/YoutubeUpload";

function App() {
	let baseurl = "";

	if (process.env.REACT_APP_ENV === "production") {
		baseurl = process.env.REACT_APP_BASE_URL_PRODUCTION;
	}
	if (process.env.REACT_APP_ENV === "development") {
		baseurl = process.env.REACT_APP_BASE_URL_DEVELOPMENT;
	}

	console.log(baseurl);

	const [blob, setBlob] = useState();
	const { status, startRecording, stopRecording, mediaBlobUrl } =
		useReactMediaRecorder({ video: false });
	const [file, setFile] = useState();
	const [emotionsData, setEmotionsData] = useState("");
	const [sampleUploaded, setSampleUploaded] = useState(false);
	const [emotions, setEmotions] = useState("");
	const [gender, setGender] = useState("");
	const [isCheckedDensenet, setIsCheckedDensenet] = useState(true);
	const [isCheckedResnet, setIsCheckedResnet] = useState(true);
	const [isCheckedAlexnet, setIsCheckedAlexnet] = useState(true);
	const [isCheckedMobilenet, setIsCheckedMobilenet] = useState(true);
	const [showCharts, setShowCharts] = useState(true);
	const [youtubeLink, setYoutubeLink] = useState("");
	const [showLargeCharts, setShowLargeCharts] = useState(true);
	const [csv, setCsv] = useState(false);

	useEffect(() => {
		handleCsvCreation(emotionsData, setCsv);
	}, [emotionsData]);

	// Upload recorded audio

	return (
		<div className="App">
			<div className="container">
				<div className="div-grid">
					{/*AUDIO UPLOADING RELATED FUNCTIONS*/}
					<Recording
						status={status}
						startRecording={startRecording}
						stopRecording={stopRecording}
						mediaBlobUrl={mediaBlobUrl}
						baseurl={baseurl}
						setSampleUploaded={setSampleUploaded}
						setEmotions={setEmotions}
						setGender={setGender}
						setEmotionsData={setEmotionsData}
					/>
					<UploadAudioFile
						baseurl={baseurl}
						setSampleUploaded={setSampleUploaded}
						setEmotions={setEmotions}
						setGender={setGender}
						setEmotionsData={setEmotionsData}
						file={file}
						setFile={setFile}
					/>
					<YoutubeUpload
						baseurl={baseurl}
						setSampleUploaded={setSampleUploaded}
						setEmotions={setEmotions}
						setGender={setGender}
						setEmotionsData={setEmotionsData}
						file={file}
						setFile={setFile}
						youtubeLink={youtubeLink}
						setYoutubeLink={setYoutubeLink}
					/>
					<SampleUploaded sampleUploaded={sampleUploaded} />
					{/* Settings div */}
					<Settings
						isCheckedDensenet={isCheckedDensenet}
						setIsCheckedDensenet={setIsCheckedDensenet}
						isCheckedResnet={isCheckedResnet}
						setIsCheckedResnet={setIsCheckedResnet}
						isCheckedAlexnet={isCheckedAlexnet}
						setIsCheckedAlexnet={setIsCheckedAlexnet}
						isCheckedMobilenet={isCheckedMobilenet}
						setIsCheckedMobilenet={setIsCheckedMobilenet}
						showCharts={showCharts}
						setShowCharts={setShowCharts}
						showLargeCharts={showLargeCharts}
						setShowLargeCharts={setShowLargeCharts}
						csv={csv}
						setCsv={setCsv}
						CSVLink={CSVLink}
					/>
					<div className="div-separate"></div>
					<div
						className={
							showLargeCharts
								? "div-classification-results"
								: "div-classification-results-small"
						}
					>
						{/* Graphs */}

						<ChartCard
							className="div-chart div-chart-dense"
							header={"DenseNet-121"}
							emotionsData={gender[0]}
							ischecked={isCheckedDensenet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Gender classification"
							classes="gender"
						/>
						<ChartCard
							className="div-chart div-chart-dense"
							header={"DenseNet-121"}
							emotionsData={emotions[0]}
							ischecked={isCheckedDensenet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Emotion classification"
							classes="emotion"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"ResNet50"}
							emotionsData={gender[1]}
							ischecked={isCheckedResnet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Gender classification"
							classes="gender"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"ResNet50"}
							emotionsData={emotions[1]}
							ischecked={isCheckedResnet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Emotion classification"
							classes="emotion"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"AlexNet"}
							emotionsData={gender[2]}
							ischecked={isCheckedAlexnet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Gender classification"
							classes="gender"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"AlexNet"}
							emotionsData={emotions[2]}
							ischecked={isCheckedAlexnet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Emotion classification"
							classes="emotion"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"MobileNetV3"}
							emotionsData={gender[3]}
							ischecked={isCheckedMobilenet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Gender classification"
							classes="gender"
						/>
						<ChartCard
							className="div-chart div-chart-resnet"
							header={"MobileNetV3"}
							emotionsData={emotions[3]}
							ischecked={isCheckedMobilenet}
							showCharts={showCharts}
							showLargeCharts={showLargeCharts}
							chartLabel="Emotion classification"
							classes="emotion"
						/>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
