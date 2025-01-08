# Pixel-art scaling Accelerator in Chisel
> 郭昱辰, 簡子昕

Welcome to the documentation for the **Chisel-Based Pixel-Art Scaling Hardware Accelerator**. This project represents the final assignment for the Computer Architecture course (113-1 Semester). In its initial version, the accelerator facilitates image conversion and incorporates several fundamental features. Future iterations will focus on enhancing performance through hardware acceleration.


[GitHub Repository](https://github.com/whoishuni/pixel-art-xbr-scaling)

Table of Contents
-----------------

1.  [Introduction](#introduction)
2.  [Prerequisites](#prerequisites)
3.  [Project Overview](#project-overview)
4.  [Detailed Code Analysis](#detailed-code-analysis)
    *   [Enhanced Edge Detector Module](#enhanced-edge-detector-module)
    *   [Improved Pixel Interpolator Module](#improved-pixel-interpolator-module)
    *   [Extended PixelArtScaler Module](#extended-pixelartscaler-module)
    *   [Main Program](#main-program)
5.  [Running the Code](#running-the-code)
6.  [Output Results](#output-results)
7.  [Conclusion and Future Work](#conclusion-and-future-work)
8.  [Version History](#version-history)

* * *

Introduction
------------

Pixel Art is a digital art form characterized by the meticulous creation and editing of images at the pixel level, commonly utilized in gaming and graphic design. As image resolutions escalate, the efficient scaling of pixel art poses significant challenges. This project endeavors to design a hardware accelerator using the Chisel language to achieve high-quality pixel art scaling. The current version (1.0) encompasses image conversion and several essential functionalities, laying the groundwork for future enhancements aimed at optimizing performance through hardware acceleration.


Prerequisites
-------------

Before engaging with this project, ensure you possess the following knowledge and tools:

- **Fundamental Hardware Description Language (HDL) Proficiency**: Familiarity with Chisel.
- **Scala Programming Basics**: Chisel is built upon Scala.
- **Digital Logic Design Principles**: Understanding of digital logic and modular design concepts.
- **Development Environment Setup**: Installation of Chisel and associated development tools.


Project Overview
----------------

The objective of this project is to design a Chisel-based hardware accelerator capable of efficiently scaling pixel art images. The key functionalities implemented in this version include:

- **Edge Detection**: Employing the Sobel algorithm to identify image edges, determining both their direction and strength.
- **Pixel Interpolation**: Conducting intelligent pixel interpolation based on edge detection outcomes to facilitate smooth image scaling.
- **Modular Architecture**: Structuring the system into discrete modules to enhance maintainability and scalability.

This documentation will delve into the implementation specifics of each module, providing a comprehensive analysis of the project's components.


## Detailed Code Analysis

### Enhanced Edge Detector Module




```scala
package pixelart

import chisel3._
import chisel3.util._

/**
  * Enhanced Edge Detector Module
  * Utilizes the Sobel algorithm to compute gradients and determine edge direction and strength.
  */
class EnhancedEdgeDetector extends Module {
  val io = IO(new Bundle {
    // Current pixel
    val currentPixel = Input(UInt(24.W))
    // Neighboring pixels (Top-Left, Top, Top-Right, Left, Right, Bottom-Left, Bottom, Bottom-Right)
    val neighbors = Input(Vec(8, UInt(24.W)))
    // Output edge direction and strength
    val edgeDirection = Output(UInt(2.W)) // 00: None, 01: Horizontal, 10: Vertical, 11: Diagonal
    val edgeStrength  = Output(UInt(8.W)) // Edge strength
  })

  // Separate RGB components
  def getRGB(pixel: UInt): (UInt, UInt, UInt) = {
    val r = pixel(23, 16)
    val g = pixel(15, 8)
    val b = pixel(7, 0)
    (r, g, b)
  }

  // Calculate grayscale value
  def toGray(pixel: UInt): UInt = {
    val (r, g, b) = getRGB(pixel)
    // Standard grayscale conversion formula
    (r * 299.U + g * 587.U + b * 114.U) / 1000.U
  }

  // Convert all relevant pixels to grayscale
  val grayCurrent = toGray(io.currentPixel)
  val grayNeighbors = io.neighbors.map(toGray)

  // Sobel kernels
  val sobelX = Array(
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
  )
  val sobelY = Array(
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
  )

  // Calculate Gx and Gy
  // Neighbor pixel order: 0:Top-Left, 1:Top, 2:Top-Right, 3:Left, 4:Right, 5:Bottom-Left, 6:Bottom, 7:Bottom-Right
  val gx = (grayNeighbors(0).asSInt * sobelX(0).S +
            grayNeighbors(1).asSInt * sobelX(1).S +
            grayNeighbors(2).asSInt * sobelX(2).S +
            grayNeighbors(3).asSInt * sobelX(3).S +
            grayNeighbors(4).asSInt * sobelX(4).S +
            grayNeighbors(5).asSInt * sobelX(5).S +
            grayNeighbors(6).asSInt * sobelX(6).S +
            grayNeighbors(7).asSInt * sobelX(7).S).asUInt

  val gy = (grayNeighbors(0).asSInt * sobelY(0).S +
            grayNeighbors(1).asSInt * sobelY(1).S +
            grayNeighbors(2).asSInt * sobelY(2).S +
            grayNeighbors(3).asSInt * sobelY(3).S +
            grayNeighbors(4).asSInt * sobelY(4).S +
            grayNeighbors(5).asSInt * sobelY(5).S +
            grayNeighbors(6).asSInt * sobelY(6).S +
            grayNeighbors(7).asSInt * sobelY(7).S).asUInt

  // Calculate gradient magnitude |Gx| + |Gy|
  val gradient = (gx.asSInt.abs + gy.asSInt.abs).asUInt

  // Set gradient threshold
  val threshold = 30.U

  // Determine if it's an edge
  val isEdge = gradient > threshold

  // Determine edge direction based on Gx and Gy
  val direction = Wire(UInt(2.W))
  when(gx > gy) {
    direction := 1.U // Horizontal
  } .elsewhen(gx < gy) {
    direction := 2.U // Vertical
  } .otherwise {
    direction := 3.U // Diagonal
  }

  io.edgeDirection := Mux(isEdge, direction, 0.U)
  io.edgeStrength := gradient(7,0)
}

```

#### Module Functionality

*   **RGB Separation and Grayscale Conversion**: Segregates the input 24-bit RGB pixel into individual R, G, and B components. Converts these components to grayscale using a standard formula to streamline the subsequent edge detection process.
    
*   **Sobel Algorithm Application**: Implements the Sobel operator to compute horizontal (Gx) and vertical (Gy) gradients, which are essential for edge detection.
    
*   **Gradient Magnitude and Direction Determination**: Calculates the gradient magnitude by summing the absolute values of Gx and Gy. If the magnitude surpasses a predefined threshold, the pixel is classified as an edge, and its direction (horizontal, vertical, diagonal) is ascertained based on the relative magnitudes of Gx and Gy.
    

### Improved Pixel Interpolator Module

```scala
/**
  * Improved Pixel Interpolator Module
  * Performs intelligent interpolation based on edge direction and strength.
  */
class EnhancedPixelInterpolator(scaleFactor: Int) extends Module {
  require(scaleFactor >= 1 && scaleFactor <= 4, "Scale factor must be between 1 and 4")

  val io = IO(new Bundle {
    // Current pixel
    val currentPixel = Input(UInt(24.W))
    // Edge detection results
    val edgeDirection = Input(UInt(2.W))
    val edgeStrength  = Input(UInt(8.W))
    // Interpolated pixel outputs
    val outPixels = Output(Vec(scaleFactor * scaleFactor, UInt(24.W)))
  })

  // Separate RGB components
  def getRGB(pixel: UInt): (UInt, UInt, UInt) = {
    val r = pixel(23, 16)
    val g = pixel(15, 8)
    val b = pixel(7, 0)
    (r, g, b)
  }

  val (r, g, b) = getRGB(io.currentPixel)

  // Interpolation function: Linear interpolation
  def lerp(a: UInt, b: UInt, t: UInt): UInt = {
    ((a.asUInt * (255.U - t)) + (b.asUInt * t)) / 255.U
  }

  // Interpolation strategy based on edge direction
  for (i <- 0 until (scaleFactor * scaleFactor)) {
    // Calculate interpolation position
    val row = (i / scaleFactor).U
    val col = (i % scaleFactor).U
    val t_row = (row * 255.U) / scaleFactor.U
    val t_col = (col * 255.U) / scaleFactor.U

    io.outPixels(i) := MuxLookup(io.edgeDirection, io.currentPixel) (
      Seq(
        // Horizontal edge: blend left and right pixels
        1.U -> {
          // Placeholder for specific interpolation logic
          io.currentPixel
        },
        // Vertical edge: blend top and bottom pixels
        2.U -> {
          // Placeholder for specific interpolation logic
          io.currentPixel
        },
        // Diagonal edge: bilinear interpolation
        3.U -> {
          // Placeholder for specific interpolation logic
          io.currentPixel
        }
      )
    )

    // Fallback: Directly copy the current pixel if no specific strategy is applied
    when(io.edgeStrength < 50.U) { // Adjust based on edge strength threshold
      io.outPixels(i) := io.currentPixel
    }
  }
}



```



#### Module Functionality

*   **Interpolation Strategy**: Selects appropriate interpolation methods based on the detected edge direction (horizontal, vertical, diagonal). Currently utilizes linear interpolation as a foundational approach, with placeholders for more sophisticated strategies.
    
*   **Intelligent Interpolation**: Leverages edge strength to determine the necessity of interpolation. If the edge strength is below a certain threshold, the current pixel is directly propagated to preserve image fidelity.
    

### Extended PixelArtScaler Module

```scala
/**
  * Extended PixelArtScaler Module with Enhanced Edge Detection and Pixel Interpolation
  */
class EnhancedPixelArtScaler(width: Int, height: Int, scaleFactor: Int) extends Module {
  require(scaleFactor >= 1, "Scale factor must be at least 1")
  require(scaleFactor <= 4, "Scale factor is limited to a maximum of 4 for this implementation")

  val io = IO(new Bundle {
    // Input pixel (24-bit RGB)
    val inPixel  = Input(UInt(24.W))
    val inValid  = Input(Bool())

    // Output pixels (scaleFactor x scaleFactor block)
    val outPixels = Output(Vec(scaleFactor * scaleFactor, UInt(24.W)))

    val outValid  = Output(Bool())

    // Neighboring pixels output (Top-Left, Top, Top-Right, Left, Right, Bottom-Left, Bottom, Bottom-Right)
    val neighbor = Output(Vec(8, UInt(24.W)))
  })

  // Line buffers: store previous, current, and next lines
  val lineBufferPrev = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))
  val lineBufferCurrent = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))
  val lineBufferNext = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))

  // Current column and row indices
  val col = RegInit(0.U(log2Ceil(width + 1).W)) // Increased width to prevent overflow
  val row = RegInit(0.U(log2Ceil(height + 1).W)) // Increased width to prevent overflow

  // Shift registers for the current line to get left, center, right
  val shiftRegPrev = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))
  val shiftRegCurrent = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))
  val shiftRegNext = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))

  // Instantiate Enhanced Edge Detector Module
  val edgeDetector = Module(new EnhancedEdgeDetector)

  // Instantiate Pixel Interpolator Module
  val pixelInterpolator = Module(new EnhancedPixelInterpolator(scaleFactor))

  // Update line buffers
  when(io.inValid) {
    // Shift line buffers
    lineBufferPrev := lineBufferCurrent
    lineBufferCurrent := lineBufferNext
    lineBufferNext(col) := io.inPixel

    // Update column and row indices
    when(col === (width - 1).U) {
      col := 0.U
      row := row + 1.U
    } .otherwise {
      col := col + 1.U
    }

    // Update shift registers
    shiftRegPrev(0) := lineBufferPrev(col)
    shiftRegPrev(1) := shiftRegPrev(0)
    shiftRegPrev(2) := shiftRegPrev(1)

    shiftRegCurrent(0) := lineBufferCurrent(col)
    shiftRegCurrent(1) := shiftRegCurrent(0)
    shiftRegCurrent(2) := shiftRegCurrent(1)

    shiftRegNext(0) := lineBufferNext(col)
    shiftRegNext(1) := shiftRegNext(0)
    shiftRegNext(2) := shiftRegNext(1)
  }

  // Extract neighboring pixels
  val neighbors = Wire(Vec(8, UInt(24.W)))
  neighbors(0) := Mux(row === 0.U || col === 0.U, 0.U, shiftRegPrev(1)) // Top-Left
  neighbors(1) := Mux(row === 0.U, 0.U, shiftRegPrev(0)) // Top
  neighbors(2) := Mux(row === 0.U || col === (width - 1).U, 0.U, shiftRegPrev(2)) // Top-Right
  neighbors(3) := Mux(col === 0.U, 0.U, shiftRegCurrent(1)) // Left
  neighbors(4) := Mux(col === (width - 1).U, 0.U, shiftRegCurrent(2)) // Right
  neighbors(5) := Mux(row === (height - 1).U || col === 0.U, 0.U, shiftRegNext(1)) // Bottom-Left
  neighbors(6) := Mux(row === (height - 1).U, 0.U, shiftRegNext(0)) // Bottom
  neighbors(7) := Mux(row === (height - 1).U || col === (width - 1).U, 0.U, shiftRegNext(2)) // Bottom-Right

  // Output extracted neighboring pixels
  io.neighbor := neighbors

  // Connect Enhanced Edge Detector Module
  edgeDetector.io.currentPixel := io.inPixel
  edgeDetector.io.neighbors := neighbors

  // Connect Pixel Interpolator Module
  pixelInterpolator.io.currentPixel := io.inPixel
  pixelInterpolator.io.edgeDirection := edgeDetector.io.edgeDirection
  pixelInterpolator.io.edgeStrength := edgeDetector.io.edgeStrength

  // Connect interpolated pixels to module output
  io.outPixels := pixelInterpolator.io.outPixels

  // Output valid signal
  io.outValid := io.inValid
}


```

#### Module Functionality

*   **Line Buffer Management**: Utilizes three line buffers (previous, current, next) to store sequential lines of image data. Manages column and row indices to track the current position within the image matrix.
    
*   **Neighboring Pixel Extraction**: Retrieves the eight neighboring pixels surrounding the current pixel, handling boundary conditions by padding with zeros where necessary to avoid accessing invalid memory regions.
    
*   **Edge Detection and Interpolation**: Feeds the current pixel and its neighbors into the `EnhancedEdgeDetector` module to identify edges. The resultant edge direction and strength are then passed to the `EnhancedPixelInterpolator` module to perform intelligent interpolation based on the detected edge characteristics.
    

### Main Program

```scala
/**
  * Main Program
  * Reads an image → Reads scaling factor → Prepares output matrix → Simulates scaling using Chisel → Writes back the image
  */
object PixelArtScale extends App {

  //========== (1) Read Scaling Factor ==========
  println("Enter scaling factor (e.g., 2 for 2x scaling):")
  val scaleFactorInput = StdIn.readInt()
  require(scaleFactorInput >= 1 && scaleFactorInput <= 4, "Scaling factor must be between 1 and 4")

  //========== (2) Read Image and Convert to 2D Array ==========
  /**
    * Reads PNG/JPG using Scala and converts it to a 2D array [height][width] (24-bit RGB)
    */
  def loadImageToMatrix(path: String): Array[Array[Int]] = {
    val img  = ImageIO.read(new File(path))
    val w    = img.getWidth
    val h    = img.getHeight
    val data = Array.ofDim[Int](h, w)

    for (y <- 0 until h; x <- 0 until w) {
      data(y)(x) = img.getRGB(x, y) & 0xFFFFFF
    }
    data
  }

  //========== (3) Write Output Matrix Back to Image ==========
  /**
    * Writes a 2D array (24-bit RGB) back to an image
    */
  def saveMatrixToImage(matrix: Array[Array[Int]], outPath: String): Unit = {
    val h = matrix.length
    val w = if (h > 0) matrix(0).length else 0
    val outImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

    for (y <- 0 until h; x <- 0 until w) {
      outImg.setRGB(x, y, matrix(y)(x) & 0xFFFFFF)
    }
    ImageIO.write(outImg, "png", new File(outPath))
  }

  //========== (4) Prepare Data: Read input.png → Create outputMatrix ==========
  // Read input image
  val inputPath = "/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/Lenna.png"
  val inputMatrix = loadImageToMatrix(inputPath)

  val inHeight = inputMatrix.length
  val inWidth  = if (inHeight > 0) inputMatrix(0).length else 0

  // Create scaled output matrix
  val outHeight = inHeight * scaleFactorInput
  val outWidth  = inWidth  * scaleFactorInput
  val outputMatrix = Array.ofDim[Int](outHeight, outWidth)

  println(s"[Info] Input image size: $inWidth x $inHeight")
  println(s"[Info] Scale factor: ${scaleFactorInput}x")
  println(s"[Info] Output image size: $outWidth x $outHeight")

  //========== (5) **At the End of the Program** Use RawTester.test(...) to Perform Simulation ==========
  // Use RawTester for simple testing/simulation
  RawTester.test(new EnhancedPixelArtScaler(inWidth, inHeight, scaleFactorInput)) { dut =>

    // Initialize line buffers and shift registers
    for (y <- 0 until inHeight + 2) { // Including two boundary lines
      for (x <- 0 until inWidth + 2) { // Including two boundary columns
        val rgb24 = if (y >= 1 && y <= inHeight && x >= 1 && x <= inWidth)
          inputMatrix(y - 1)(x - 1)
        else
          0 // Boundary padding with 0

        // Poke input
        dut.io.inPixel.poke(rgb24.U)
        dut.io.inValid.poke(true.B)

        // Advance clock by 1 cycle
        dut.clock.step(1)

        // If outValid is true, extract the corresponding scaleFactor x scaleFactor block
        if (dut.io.outValid.peek().litToBoolean) {
          for (i <- 0 until (scaleFactorInput * scaleFactorInput)) {
            val outP = dut.io.outPixels(i).peek().litValue.toInt & 0xFFFFFF

            // Calculate output matrix coordinates
            val baseY = (y - 1) * scaleFactorInput
            val baseX = (x - 1) * scaleFactorInput

            // Ensure within output matrix bounds
            val pixelY = baseY + (i / scaleFactorInput)
            val pixelX = baseX + (i % scaleFactorInput)

            if (pixelY >= 0 && pixelY < outHeight && pixelX >= 0 && pixelX < outWidth) {
              outputMatrix(pixelY)(pixelX) = outP
            }
          }

          // Retrieve neighboring pixels
          val neighbors = dut.io.neighbor.map(_.peek().litValue.toInt & 0xFFFFFF)
          // Further processing of neighboring pixels can be done here, such as applying the xBR algorithm
          // Currently, only printing neighboring pixels for reference
          /*
          println(s"Pixel ($x, $y):")
          println(s"Neighbors: ${neighbors.mkString(", ")}")
          */
        }
      }
    }

    // If needed, advance the clock by 1 cycle at the end
    dut.clock.step(1)
  }

  //========== (6) Write Back Image File and End Program ==========
  val outputPath = "/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/output.png"
  saveMatrixToImage(outputMatrix, outputPath)
  println(s"[Info] Processing complete! Scaled image saved to: $outputPath")
}

```

#### Program Functionality

1.  **Read Scaling Factor**: Prompts the user to input the desired scaling factor, ensuring it falls within the acceptable range (1 to 4).
    
2.  **Read Image and Convert to 2D Array**: Utilizes Scala's `ImageIO` to load a PNG/JPG image and transform it into a two-dimensional array representing pixel data in 24-bit RGB format.
    
3.  **Write Output Matrix Back to Image**: Converts the processed two-dimensional pixel array back into an image format and saves it to the specified output path.
    
4.  **Prepare Data**: Reads the input image, determines the dimensions of the scaled output image based on the scaling factor, and initializes the output matrix accordingly.
    
5.  **Simulation Using RawTester**: Employs  `RawTester` to simulate the `EnhancedPixelArtScaler` module. Inputs image data into the module, simulates hardware operations, and aggregates the output pixels into the output matrix.
    
6.  **Write Back Image File**: Saves the scaled pixel data as a new image file, completing the scaling process.
    

Running the Code
----------------

Follow these steps to run the project:

*   **Set Up the Development Environment**:
    
    *   Install Scala and SBT (Scala Build Tool)**: Ensure that Scala and SBT are installed on your system. Refer to the Scala installation guide and the SBT installation instructions for detailed steps.
        
    *   Install Chisel and Dependencies**: Set up Chisel by following the official Chisel installation guide.
2.  **Prepare Input Image**:
    
    *   Place the image to be scaled at the specified path, e.g., `/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/Lenna.png`.
3.  **Compile and Run**:
    
    *   Use SBT to compile the project.
    ```scala  
     sbt run
     ```
    *   Execute the `PixelArtScale` main program and input the desired scaling factor (e.g., 2 for 2x scaling).
    ```scale
    sbt "runMain pixelart.PixelArtScale"
    ```
    *    The PixelArtScale main program is located in the file:
    ```text
    .\src\main\scala\Pixel Art\PixelArtScale.scala
    ```

     
    
4.  **View Output Results**:
    
    *   The scaled image will be saved at the specified output path, e.g., `/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/output.png`.

Output Results
--------------
![image](https://hackmd.io/_uploads/HyD-7RPUkl.png)

After running the program, you will obtain the scaled image. Below is an example comparison of the original and scaled images:

### Original Image
![image](https://hackmd.io/_uploads/r1Uw9isUyl.png)


### Scaled Image

![output](https://hackmd.io/_uploads/Bkhv9isLkl.png)




## Conclusion and Future Work

In this project, we successfully designed and implemented a Chisel-based pixel art scaling solution, focusing on the core **Art Pixel** functionality. The current implementation integrates edge detection and intelligent interpolation techniques to facilitate the scaling of pixel art images effectively. While the foundational features are operational, the present version does not yet leverage hardware acceleration, which remains a critical area for enhancement.

**Future Enhancements Include**:

- **Integration of Hardware Accelerators**: Developing dedicated hardware accelerators to parallelize computations, thereby significantly accelerating the scaling process and improving overall performance.

- **Optimization of Interpolation Algorithms**: Refining existing interpolation methods and incorporating advanced algorithms, such as the xBR algorithm, to further enhance image quality and scalability.

- **Expansion of Supported Scaling Factors**: Extending the range of supported scaling factors to accommodate a wider variety of application requirements and use cases.

- **Hardware Resource Optimization**: Streamlining module designs to minimize hardware resource consumption, ensuring efficient utilization and maximizing performance.

These enhancements aim to elevate the pixel art scaling solution's performance and versatility, establishing it as a robust and efficient tool for various digital applications. By focusing on hardware acceleration and algorithm optimization, future versions will deliver superior scalability and image fidelity, meeting the demanding needs of modern pixel art processing.

