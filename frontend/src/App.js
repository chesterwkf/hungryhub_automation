import React, { useState } from "react";
import { Container, Row, Col, Button, Card, Form } from "react-bootstrap";

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [restaurantName, setRestaurantName] = useState("");

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

    const formData = new FormData();
    formData.append("restaurantName", restaurantName); // Add restaurant name to form data
    selectedFiles.forEach((file) => formData.append("images", file));

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      alert(data.message);
    } catch (error) {
      console.error("Error uploading images:", error);
    }
  };

  return (
    <Container className="mt-5">
      <h1 className="text-center">HungryHub Automated Onboarding</h1>
      <p className="text-center text-muted">
        Enter the restaurant name and upload all the menu images before pressing the "Upload Images" button.
      </p>

      {/* Restaurant Name Input */}
      <Form.Group className="mb-3" controlId="restaurantName">
        <Form.Label>Restaurant Name</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter restaurant name"
          value={restaurantName}
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
          className="form-control"
          style={{ maxWidth: "400px", margin: "0 auto" }}
        />
      </div>

      {/* Upload Button */}
      <div className="text-center mb-4">
        <Button variant="primary" onClick={handleUpload}>
          Upload Images
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
