import React, { useState } from "react";
import {
  Container,
  Row,
  Col,
  Button,
  Card,
  Form,
  Spinner,
  Alert,
} from "react-bootstrap";

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [restaurantName, setRestaurantName] = useState("");
  const [averagePrice, setAveragePrice] = useState("");
  const [categorizedMenuText, setCategorizedMenuText] = useState("");
  const [editedMenuText, setEditedMenuText] = useState("");
  const [isEditing, setIsEditing] = useState(false);

  // State to track processing stages
  const [isProcessingMenuItems, setIsProcessingMenuItems] = useState(false);
  const [isProcessingExcel, setIsProcessingExcel] = useState(false);
  const [isSavingEdits, setIsSavingEdits] = useState(false);

  // State to track workflow progression
  const [menuItemsProcessed, setMenuItemsProcessed] = useState(false);

  // Error/success state
  const [statusMessage, setStatusMessage] = useState({
    type: null,
    text: null,
  });

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    const filePreviews = files.map((file) => URL.createObjectURL(file));
    setSelectedFiles(files);
    setPreviews(filePreviews);
    // Reset process state if files change
    setMenuItemsProcessed(false);
    setCategorizedMenuText("");
    setStatusMessage({ type: null, text: null });
  };

  // STEP 1: Upload images, process menu, and show categorized items
  const handleProcessMenuItems = async () => {
    // Validate inputs
    if (!restaurantName.trim()) {
      setStatusMessage({
        type: "danger",
        text: "Please enter the restaurant name!",
      });
      return;
    }
    if (!averagePrice.trim()) {
      setStatusMessage({
        type: "danger",
        text: "Please enter the average price!",
      });
      return;
    }
    if (selectedFiles.length === 0) {
      setStatusMessage({
        type: "danger",
        text: "Please select images to upload!",
      });
      return;
    }

    // Start processing
    setIsProcessingMenuItems(true);
    setStatusMessage({
      type: "info",
      text: "Uploading images and processing menu items...",
    });
    setCategorizedMenuText("");
    setMenuItemsProcessed(false);

    try {
      // 1. Upload images
      const formData = new FormData();
      formData.append("restaurantName", restaurantName);
      selectedFiles.forEach((file) => formData.append("images", file));

      const uploadResponse = await fetch("http://localhost:5000/api/upload", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadResponse.json();
      if (uploadData.error) throw new Error(uploadData.error);

      const uploadData = await uploadResponse.json();
      if (uploadData.error) throw new Error(uploadData.error);

      setStatusMessage({
        type: "info",
        text: "Images uploaded. Processing menu items...",
      });
      const processData = await processResponse.json();
      if (processData.error) throw new Error(processData.error);

      // 2. Process menu to extract items
      const processResponse = await fetch(
        "http://localhost:5000/api/process-menu",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ restaurantName }),
        }
      );

      const processData = await processResponse.json();
      if (processData.error)
        throw new Error(`Processing menu failed: ${processData.error}`);

      setStatusMessage({
        type: "info",
        text: "Menu items processed. Fetching categorized menu...",
      });

      // 3. Fetch and display the categorized menu text
      const menuTextResponse = await fetch(
        "http://localhost:5000/api/get-categorized-menu",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ restaurantName }),
        }
      );

      const menuTextData = await menuTextResponse.json();
      if (menuTextData.error)
        throw new Error(`Failed to fetch menu text: ${menuTextData.error}`);

      setCategorizedMenuText(menuTextData.menuText);
      setEditedMenuText(menuTextData.menuText); // Initialize edited text with original text
      setMenuItemsProcessed(true);
      setStatusMessage({
        type: "success",
        text: "Menu items processed successfully! You can review and edit them below, then click 'Generate Excel' to continue.",
      });
    } catch (error) {
      console.error("Error processing menu items:", error);
      setStatusMessage({ type: "danger", text: `Error: ${error.message}` });
      setMenuItemsProcessed(false);
      setCategorizedMenuText("");
    } finally {
      setIsProcessingMenuItems(false);
    }
  };

  // Toggle edit mode
  const toggleEditMode = () => {
    if (isEditing) {
      // If we're exiting edit mode, save changes
      handleSaveEdits();
    } else {
      // If we're entering edit mode, just toggle the state
      setIsEditing(true);
    }
  };

  // Save edited menu text
  const handleSaveEdits = async () => {
    setIsSavingEdits(true);
    try {
      const response = await fetch(
        "http://localhost:5000/api/update-categorized-menu",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            restaurantName,
            menuText: editedMenuText,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to save menu edits");
      }

      setCategorizedMenuText(editedMenuText);
      setIsEditing(false);
      setStatusMessage({
        type: "success",
        text: "Menu edits saved successfully!",
      });
    } catch (error) {
      console.error("Error saving menu edits:", error);
      setStatusMessage({
        type: "danger",
        text: `Error saving edits: ${error.message}`,
      });
    } finally {
      setIsSavingEdits(false);
    }
  };

  // Handle text changes in the editor
  const handleMenuTextChange = (event) => {
    setEditedMenuText(event.target.value);
  };

  // STEP 2: Generate bundles and download Excel
  const handleGenerateExcel = async () => {
    // If there are unsaved changes, save them first
    if (isEditing) {
      await handleSaveEdits();
    }

    setIsProcessingExcel(true);
    setStatusMessage({
      type: "info",
      text: "Generating bundles and creating Excel file...",
    });

    try {
      // 1. Generate bundles
      const bundleResponse = await fetch(
        "http://localhost:5000/api/generate-bundles",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ restaurantName, averagePrice }),
        }
      );

      const bundleData = await bundleResponse.json();
      if (!bundleResponse.ok || bundleData.error) {
        throw new Error(
          bundleData.error ||
            `Bundle generation failed: ${bundleResponse.statusText}`
        );
      }

      setStatusMessage({
        type: "info",
        text: "Bundles generated. Creating and downloading Excel file...",
      });

      // 2. Generate and download Excel proposal
      const proposalResponse = await fetch(
        "http://localhost:5000/api/generate-proposal",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ restaurantName }),
        }
      );

      if (!proposalResponse.ok) {
        const errorText = await proposalResponse.text();
        throw new Error(`Excel generation failed: ${errorText}`);
      }

      // 3. Download the Excel file
      const blob = await proposalResponse.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute(
        "download",
        `${restaurantName || "restaurant"}_proposal.xlsx`
      );
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

      setStatusMessage({
        type: "success",
        text: "Excel file downloaded successfully!",
      });

      // Reset workflow to start fresh
      setMenuItemsProcessed(false);
      setCategorizedMenuText("");
    } catch (error) {
      console.error("Error generating Excel:", error);
      setStatusMessage({
        type: "danger",
        text: `Error generating Excel: ${error.message}`,
      });
    } finally {
      setIsProcessingExcel(false);
    }
  };

  // Determine if inputs should be disabled
  const inputsDisabled =
    isProcessingMenuItems || isProcessingExcel || menuItemsProcessed;

  return (
    <Container className="mt-5">
      <h1 className="text-center">HungryHub Automated Onboarding</h1>
      <p className="text-center text-muted">
        Enter restaurant details and upload menu images to begin the process.
      </p>

      {statusMessage.type && (
        <Alert
          variant={statusMessage.type}
          onClose={() => setStatusMessage({ type: null, text: null })}
          dismissible
        >
          {statusMessage.text}
        </Alert>
      )}

      {/* Restaurant Name Input */}
      <Form.Group className="mb-3" controlId="restaurantName">
        <Form.Label>Restaurant Name</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter restaurant name"
          value={restaurantName}
          disabled={inputsDisabled}
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
          disabled={inputsDisabled}
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
          disabled={inputsDisabled}
          className="form-control"
          style={{ maxWidth: "400px", margin: "0 auto" }}
        />
      </div>

      {/* Action Button - changes based on workflow stage */}
      <div className="text-center mb-4">
        {!menuItemsProcessed ? (
          <Button
            variant="primary"
            onClick={handleProcessMenuItems}
            disabled={isProcessingMenuItems}
          >
            {isProcessingMenuItems ? (
              <>
                <Spinner animation="border" size="sm" /> Processing Menu
                Items...
              </>
            ) : (
              "Process Menu Items"
            )}
          </Button>
        ) : (
          <Button
            variant="success"
            onClick={handleGenerateExcel}
            disabled={isProcessingExcel}
            className="mt-3"
          >
            {isProcessingExcel ? (
              <>
                <Spinner animation="border" size="sm" /> Generating Excel...
              </>
            ) : (
              "Generate Excel"
            )}
          </Button>
        )}
      </div>

      {/* Display Categorized Menu Text */}
      {categorizedMenuText && (
        <Card className="mb-4">
          <Card.Header
            as="h5"
            className="d-flex justify-content-between align-items-center"
          >
            <div>Categorized Menu Items</div>
            <Button
              variant={isEditing ? "success" : "primary"}
              size="sm"
              onClick={toggleEditMode}
              disabled={isSavingEdits}
            >
              {isSavingEdits ? (
                <>
                  <Spinner animation="border" size="sm" /> Saving...
                </>
              ) : isEditing ? (
                "Save Changes"
              ) : (
                "Edit Menu"
              )}
            </Button>
          </Card.Header>
          <Card.Body>
            {isEditing ? (
              <Form.Control
                as="textarea"
                value={editedMenuText}
                onChange={handleMenuTextChange}
                style={{
                  minHeight: "400px",
                  fontFamily: "monospace",
                  fontSize: "0.9rem",
                }}
              />
            ) : (
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: "400px",
                  overflowY: "auto",
                  fontSize: "0.9rem",
                  backgroundColor: "#f8f9fa",
                  padding: "15px",
                  borderRadius: "5px",
                }}
              >
                {categorizedMenuText}
              </pre>
            )}
            {menuItemsProcessed && (
              <div className="text-center mt-3">
                <Button
                  variant="success"
                  onClick={handleGenerateExcel}
                  disabled={isProcessingExcel || isSavingEdits}
                >
                  {isProcessingExcel ? (
                    <>
                      <Spinner animation="border" size="sm" /> Generating
                      Excel...
                    </>
                  ) : (
                    "Generate Excel"
                  )}
                </Button>
              </div>
            )}
          </Card.Body>
        </Card>
      )}

      {/* Image Previews - only show before processing is complete */}
      {previews.length > 0 && !menuItemsProcessed && (
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
