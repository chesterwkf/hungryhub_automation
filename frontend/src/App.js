import React, { useState } from "react";
import { Container, Row, Col, Button, Card, Form, Spinner, Alert } from "react-bootstrap";

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [restaurantName, setRestaurantName] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [excelReady, setExcelReady] = useState(false);
  const [averagePrice, setAveragePrice] = useState("");


  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);

    // Generate previews for selected images
    const filePreviews = files.map((file) => URL.createObjectURL(file));
    setSelectedFiles(files);
    setPreviews(filePreviews);
  };

  // Upload images and process workflow
  const handleUploadAndProcess = async () => {
    if (!restaurantName.trim()) {
      alert("Please enter the restaurant name!");
      return;
    }
    if (!averagePrice.trim()) {
      alert("Please enter the average price!");
      return;
    }    
    if (selectedFiles.length === 0) {
      alert("Please select images to upload!");
      return;
    }

    setIsProcessing(true);
    setIsSuccess(false);
    setExcelReady(false);

    // Upload images
    const formData = new FormData();
    formData.append("restaurantName", restaurantName);
    selectedFiles.forEach((file) => formData.append("images", file));
    try {
      const uploadResponse = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadResponse.json();
      if (uploadData.error) throw new Error(uploadData.error);

      // Process images
      const processResponse = await fetch("http://localhost:5000/api/process-menu", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ restaurantName }),
      });
      const processData = await processResponse.json();
      if (processData.error) throw new Error(processData.error);

      // Generate bundles
      const bundleResponse = await fetch("http://localhost:5000/api/generate-bundles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ restaurantName }),
      });
      const bundleData = await bundleResponse.json();
      if (bundleData.error) throw new Error(bundleData.error);

      // Excel is now ready to download
      setExcelReady(true);
      setIsProcessing(false);
      setIsSuccess(true);
    } catch (error) {
      alert(`Error: ${error.message}`);
      setIsProcessing(false);
    }
  };

  // Download Excel file
  const handleDownloadExcel = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/generate-proposal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ restaurantName }),
      });
      if (!response.ok) {
        let errorMsg = response.statusText;
        try {
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch { }
        throw new Error(errorMsg);
      }
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', `${restaurantName || 'restaurant'}_proposal.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
      setIsProcessing(false);
      setExcelReady(false); // Hide download button after download
      setIsSuccess(false);
    } catch (error) {
      alert(`Error downloading excel: ${error.message}`);
      setIsProcessing(false);
    }
  };

  return (
    <Container className="mt-5">
      <h1 className="text-center">HungryHub Automated Onboarding</h1>
      <p className="text-center text-muted">
        Enter the restaurant name and upload all the menu images before pressing the "Process Images" button.
      </p>

      {isProcessing && (
        <div className="text-center mb-3">
          <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          Processing images, please wait...
        </div>
      )}
      {isSuccess && (
        <Alert variant="success" className="text-center">
          All images processed! You can now download the Excel proposal.
        </Alert>
      )}

      {/* Restaurant Name Input */}
      <Form.Group className="mb-3" controlId="restaurantName">
        <Form.Label>Restaurant Name</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter restaurant name"
          value={restaurantName}
          disabled={isProcessing || excelReady}
          onChange={(e) => setRestaurantName(e.target.value)}
        />
      </Form.Group>

      <Form.Group className="mb-3" controlId="averagePrice">
        <Form.Label>Average Price</Form.Label>
        <Form.Control
          type="number"
          min="0"
          step="0.01"
          placeholder="Enter average price"
          value={averagePrice}
          disabled={isProcessing || excelReady}
          onChange={(e) => setAveragePrice(e.target.value)}
        />
      </Form.Group>

      {/* File Input */}
      <div className="text-center mb-3">
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileChange}
          disabled={isProcessing || excelReady}
          className="form-control"
          style={{ maxWidth: "400px", margin: "0 auto" }}
        />
      </div>

      {/* Action Buttons */}
      <div className="text-center mb-4">
        {!excelReady ? (
          <Button
            variant="primary"
            onClick={handleUploadAndProcess}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <Spinner animation="border" size="sm" /> Processing...
              </>
            ) : (
              "Process Images"
            )}
          </Button>
        ) : (
          <Button
            variant="success"
            onClick={handleDownloadExcel}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <Spinner animation="border" size="sm" /> Downloading...
              </>
            ) : (
              "Download Excel"
            )}
          </Button>
        )}
      </div>

      {/* Image Previews */}
      {previews.length > 0 && (
        <>
          <h3 className="text-center">Selected Images</h3>
          <Row className="mt-4">
            {previews.map((src, index) => (
              <Col xs={12} sm={6} md={4} lg={3} key={index} className="mb-4">
                <Card>
                  <Card.Img variant="top" src={src} alt={`Preview ${index}`} />
                  <Card.Body>
                    <Card.Text>Image {index + 1}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        </>
      )}
    </Container>
  );
}

export default App;
