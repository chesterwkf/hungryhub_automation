import React, { useState } from "react";
import { Container, Row, Col, Button, Card, Form, Table } from "react-bootstrap";

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [restaurantName, setRestaurantName] = useState("");
  const [menuData, setMenuData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);

    // Generate previews for selected images
    const filePreviews = files.map((file) => URL.createObjectURL(file));
    setSelectedFiles(files);
    setPreviews(filePreviews);
  };

  const handleUpload = async () => {
    if (!restaurantName.trim()) {
      alert("Please enter the restaurant name!");
      return;
    }

    if (selectedFiles.length === 0) {
      alert("Please select images to upload!");
      return;
    }

    setIsProcessing(true);
    setIsSuccess(false);

    const formData = new FormData();
    formData.append("restaurantName", restaurantName); // Add restaurant name to form data
    selectedFiles.forEach((file) => formData.append("images", file));

    try {
      const response = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        alert(`Error: ${data.error}`);
        setIsProcessing(false);
      } else {
        await handleProcessImages();
      }
    } catch (error) {
      console.error("Error uploading images:", error);
      setIsProcessing(false);
    }
  };

  const handleProcessImages = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/process-menu", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ restaurantName }),
      });

      const data = await response.json();

      if (data.error) {
        alert(`Error: ${data.error}`);
        setIsProcessing(false);
      } else {
        await handleGenerateBundles();
      }
    } catch (error) {
      console.error("Error processing images:", error);
      setIsProcessing(false);
    }
  };

  const handleGenerateBundles = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/generate-bundles", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ restaurantName }),
      });

      const data = await response.json();
      if (data.error) {
        alert(`Error generating bundles: ${data.error}`);
        setIsProcessing(false);
      } else {
        await handleGenerateExcel();
      }
    } catch (error) {
      console.error("Error generating bundles:", error);
      setIsProcessing(false);
    }
  };

  const handleGenerateExcel = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/generate-proposal", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ restaurantName }),
      });

      // Check if the response indicates success. If not, try to parse potential JSON error.
      if (!response.ok) {
        let errorMsg = response.statusText;
        try {
          // Attempt to parse error response as JSON
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch (e) {
          // Ignore error if response is not JSON
          console.log("Response was not JSON, using status text for error.");
        }
        alert(`Error generating excel: ${errorMsg}`);
        setIsProcessing(false);
        return; // Exit the function
      }

      // If response is OK, process it as a blob (file)
      const blob = await response.blob();

      // Create a URL for the blob object
      const downloadUrl = window.URL.createObjectURL(blob);

      // Create a temporary anchor element and trigger download
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', `${restaurantName || 'restaurant'}_proposal.xlsx`); // Set filename
      document.body.appendChild(link);
      link.click();

      // Clean up: remove the link and revoke the URL
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

      console.log("Generated and downloaded Excel proposal.");
      // Remove setMenuData as we are downloading the file, not storing JSON data
      // setMenuData(data);
      setIsProcessing(false);
      setIsSuccess(true);
      setTimeout(() => {
        setIsSuccess(false);
      }, 5000);

    } catch (error) {
      // Catch network errors or other issues
      console.error("Error in handleGenerateExcel:", error);
      alert(`Error generating excel: ${error.message}`);
      setIsProcessing(false);
    }
  };
  return (
    <Container className="mt-5">
      <h1 className="text-center">HungryHub Automated Onboarding</h1>
      <p className="text-center text-muted">
        Enter the restaurant name and upload all the menu images before pressing the "Upload Images" button.
      </p>

      {isProcessing && (
        <div className="text-center mb-3">
          <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          Processing images, please wait...
        </div>
      )}
      {isSuccess && (
        <div className="text-center mb-3 text-success">
          <strong>All images processed!</strong>
        </div>
      )}

      {/* Restaurant Name Input */}
      <Form.Group className="mb-3" controlId="restaurantName">
        <Form.Label>Restaurant Name</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter restaurant name"
          value={restaurantName}
          disabled={isProcessing}
          onChange={(e) => setRestaurantName(e.target.value)}
        />
      </Form.Group>

      {/* File Input */}
      <div className="text-center mb-3">
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileChange}
          disabled={isProcessing}
          className="form-control"
          style={{ maxWidth: "400px", margin: "0 auto" }}
        />
      </div>

      {/* Upload Button */}
      <div className="text-center mb-4">
        <Button variant="success" onClick={handleUpload} disabled={isProcessing} className="ms-2">
          Process Images
        </Button>
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
