package pixelart

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.RawTester

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File
import scala.io.StdIn

/**
  * Enhanced Edge Detection Module
  * Detects edges in an image using the Sobel algorithm, calculating gradients 
  * and determining edge direction and strength based on gradient magnitude and direction.
  */
class EnhancedEdgeDetector extends Module {
  val io = IO(new Bundle {
    // Input pixel (24-bit RGB)
    val currentPixel = Input(UInt(24.W))
    // Neighboring pixels (8 directions: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right)
    val neighbors = Input(Vec(8, UInt(24.W)))
    // Outputs: edge direction and strength
    val edgeDirection = Output(UInt(2.W)) // 00: None, 01: Horizontal, 10: Vertical, 11: Diagonal
    val edgeStrength  = Output(UInt(8.W)) // Edge strength
  })

  // Extract RGB components from a 24-bit pixel
  def getRGB(pixel: UInt): (UInt, UInt, UInt) = {
    val r = pixel(23, 16)
    val g = pixel(15, 8)
    val b = pixel(7, 0)
    (r, g, b)
  }

  // Convert a pixel to grayscale using the standard grayscale formula
  def toGray(pixel: UInt): UInt = {
    val (r, g, b) = getRGB(pixel)
    (r * 299.U + g * 587.U + b * 114.U) / 1000.U
  }

  // Convert the current pixel and its neighbors to grayscale
  val grayCurrent = toGray(io.currentPixel)
  val grayNeighbors = io.neighbors.map(toGray)

  // Sobel kernels for edge detection
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

  // Compute gradients Gx and Gy using Sobel kernels
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

  // Set a threshold to determine edges
  val threshold = 30.U

  // Determine if the current pixel is part of an edge
  val isEdge = gradient > threshold

  // Compute edge direction based on Gx and Gy
  val direction = Wire(UInt(2.W))
  when(gx > gy) {
    direction := 1.U // Horizontal edge
  } .elsewhen(gx < gy) {
    direction := 2.U // Vertical edge
  } .otherwise {
    direction := 3.U // Diagonal edge
  }

  // Output edge direction and strength
  io.edgeDirection := Mux(isEdge, direction, 0.U) // 0 if not an edge
  io.edgeStrength := gradient(7, 0) // Output 8-bit edge strength
}

/**
  * Enhanced Pixel Interpolator Module
  * Performs interpolation based on edge detection results.
  * Supports scaling factors of 1x to 4x.
  */
class EnhancedPixelInterpolator(scaleFactor: Int) extends Module {
  require(scaleFactor >= 1 && scaleFactor <= 4, "Scale factor must be between 1 and 4")

  val io = IO(new Bundle {
    // Input pixel (24-bit RGB)
    val currentPixel = Input(UInt(24.W))
    // Edge detection results
    val edgeDirection = Input(UInt(2.W)) // Edge direction
    val edgeStrength  = Input(UInt(8.W)) // Edge strength
    // Output pixels after interpolation (scaleFactor x scaleFactor block)
    val outPixels = Output(Vec(scaleFactor * scaleFactor, UInt(24.W)))
  })

  // Extract RGB components
  def getRGB(pixel: UInt): (UInt, UInt, UInt) = {
    val r = pixel(23, 16)
    val g = pixel(15, 8)
    val b = pixel(7, 0)
    (r, g, b)
  }

  // Linear interpolation between two values
  def lerp(a: UInt, b: UInt, t: UInt): UInt = {
    ((a.asUInt * (255.U - t)) + (b.asUInt * t)) / 255.U
  }

  // Extract RGB components from the input pixel
  val (r, g, b) = getRGB(io.currentPixel)

  // Fill output pixels based on edge direction
  for (i <- 0 until (scaleFactor * scaleFactor)) {
    val row = (i / scaleFactor).U
    val col = (i % scaleFactor).U
    val t_row = (row * 255.U) / scaleFactor.U
    val t_col = (col * 255.U) / scaleFactor.U

    io.outPixels(i) := MuxLookup(io.edgeDirection, io.currentPixel)(
      Seq(
        // Horizontal edge: Mix left and right pixels
        1.U -> io.currentPixel, // Replace with proper interpolation logic
        // Vertical edge: Mix top and bottom pixels
        2.U -> io.currentPixel, // Replace with proper interpolation logic
        // Diagonal edge: Bilinear interpolation
        3.U -> io.currentPixel  // Replace with proper interpolation logic
      )
    )
  }

  // If the edge strength is low, replicate the current pixel
  for (i <- 0 until (scaleFactor * scaleFactor)) {
    when(io.edgeStrength < 50.U) {
      io.outPixels(i) := io.currentPixel
    }
  }
}

/**
  * Enhanced Pixel Art Scaler
  * Combines edge detection and interpolation to scale images by a specified factor.
  */
class EnhancedPixelArtScaler(width: Int, height: Int, scaleFactor: Int) extends Module {
  require(scaleFactor >= 1, "Scale factor must be at least 1")
  require(scaleFactor <= 4, "Scale factor is limited to a maximum of 4 for this implementation")

  val io = IO(new Bundle {
    val inPixel  = Input(UInt(24.W))                  // Input pixel (24-bit RGB)
    val inValid  = Input(Bool())                     // Valid signal for input
    val outPixels = Output(Vec(scaleFactor * scaleFactor, UInt(24.W))) // Scaled output pixels
    val outValid  = Output(Bool())                   // Valid signal for output
    val neighbor = Output(Vec(8, UInt(24.W)))        // Neighboring pixels for edge detection
  })

  // Row buffers to store previous, current, and next rows of the image
  val lineBufferPrev = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))
  val lineBufferCurrent = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))
  val lineBufferNext = RegInit(VecInit(Seq.fill(width)(0.U(24.W))))

  // Current column and row indices
  val col = RegInit(0.U(log2Ceil(width + 1).W))
  val row = RegInit(0.U(log2Ceil(height + 1).W))

  // Shift registers for neighboring pixel access
  val shiftRegPrev = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))
  val shiftRegCurrent = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))
  val shiftRegNext = RegInit(VecInit(Seq.fill(3)(0.U(24.W))))

  // Instantiate edge detection and pixel interpolation modules
  val edgeDetector = Module(new EnhancedEdgeDetector)
  val pixelInterpolator = Module(new EnhancedPixelInterpolator(scaleFactor))

  // Update row buffers and indices on valid input
  when(io.inValid) {
    lineBufferPrev := lineBufferCurrent
    lineBufferCurrent := lineBufferNext
    lineBufferNext(col) := io.inPixel

    when(col === (width - 1).U) {
      col := 0.U
      row := row + 1.U
    } .otherwise {
      col := col + 1.U
    }

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

  // Determine neighboring pixels
  val neighbors = Wire(Vec(8, UInt(24.W)))
  neighbors(0) := Mux(row === 0.U || col === 0.U, 0.U, shiftRegPrev(1)) // Top-left
  neighbors(1) := Mux(row === 0.U, 0.U, shiftRegPrev(0))               // Top
  neighbors(2) := Mux(row === 0.U || col === (width - 1).U, 0.U, shiftRegPrev(2)) // Top-right
  neighbors(3) := Mux(col === 0.U, 0.U, shiftRegCurrent(1))            // Left
  neighbors(4) := Mux(col === (width - 1).U, 0.U, shiftRegCurrent(2)) // Right
  neighbors(5) := Mux(row === (height - 1).U || col === 0.U, 0.U, shiftRegNext(1)) // Bottom-left
  neighbors(6) := Mux(row === (height - 1).U, 0.U, shiftRegNext(0))   // Bottom
  neighbors(7) := Mux(row === (height - 1).U || col === (width - 1).U, 0.U, shiftRegNext(2)) // Bottom-right

  io.neighbor := neighbors

  // Connect edge detector
  edgeDetector.io.currentPixel := io.inPixel
  edgeDetector.io.neighbors := neighbors

  // Connect pixel interpolator
  pixelInterpolator.io.currentPixel := io.inPixel
  pixelInterpolator.io.edgeDirection := edgeDetector.io.edgeDirection
  pixelInterpolator.io.edgeStrength := edgeDetector.io.edgeStrength

  io.outPixels := pixelInterpolator.io.outPixels
  io.outValid := io.inValid
}

/**
  * Main object to handle image scaling
  * Reads input image, scales it using Chisel modules, and saves the result.
  */
object PixelArtScale extends App {
  // (1) Get scale factor from the user
  println("Enter the scaling factor (e.g., 2 for 2x scaling):")
  val scaleFactorInput = StdIn.readInt()
  require(scaleFactorInput >= 1 && scaleFactorInput <= 4, "Scale factor must be between 1 and 4")

  // (2) Load input image into a 2D array
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

  // (3) Save a 2D array as an image
  def saveMatrixToImage(matrix: Array[Array[Int]], outPath: String): Unit = {
    val h = matrix.length
    val w = if (h > 0) matrix(0).length else 0
    val outImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

    for (y <- 0 until h; x <- 0 until w) {
      outImg.setRGB(x, y, matrix(y)(x) & 0xFFFFFF)
    }
    ImageIO.write(outImg, "png", new File(outPath))
  }

  // (4) Prepare input and output matrices
  val inputPath = "/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/Lenna.png"
  val inputMatrix = loadImageToMatrix(inputPath)

  val inHeight = inputMatrix.length
  val inWidth  = if (inHeight > 0) inputMatrix(0).length else 0

  val outHeight = inHeight * scaleFactorInput
  val outWidth  = inWidth  * scaleFactorInput
  val outputMatrix = Array.ofDim[Int](outHeight, outWidth)

  println(s"[Info] Input image size: $inWidth x $inHeight")
  println(s"[Info] Scale factor: ${scaleFactorInput}x")
  println(s"[Info] Output image size: $outWidth x $outHeight")

  // (5) Use Chisel to simulate the scaling process
  RawTester.test(new EnhancedPixelArtScaler(inWidth, inHeight, scaleFactorInput)) { dut =>
    for (y <- 0 until inHeight + 2) {
      for (x <- 0 until inWidth + 2) {
        val rgb24 = if (y >= 1 && y <= inHeight && x >= 1 && x <= inWidth)
          inputMatrix(y - 1)(x - 1)
        else
          0 // Fill boundary with 0

        dut.io.inPixel.poke(rgb24.U)
        dut.io.inValid.poke(true.B)

        dut.clock.step(1)

        if (dut.io.outValid.peek().litToBoolean) {
          for (i <- 0 until (scaleFactorInput * scaleFactorInput)) {
            val outP = dut.io.outPixels(i).peek().litValue.toInt & 0xFFFFFF
            val baseY = (y - 1) * scaleFactorInput
            val baseX = (x - 1) * scaleFactorInput

            val pixelY = baseY + (i / scaleFactorInput)
            val pixelX = baseX + (i % scaleFactorInput)

            if (pixelY >= 0 && pixelY < outHeight && pixelX >= 0 && pixelX < outWidth) {
              outputMatrix(pixelY)(pixelX) = outP
            }
          }
        }
      }
    }
    dut.clock.step(1)
  }

  // (6) Save the scaled image
  val outputPath = "/home/yckuo/桌面/chisel-book/src/main/scala/Pixel-art scaling Accelerator/output.png"
  saveMatrixToImage(outputMatrix, outputPath)
  println(s"[Info] Done! Scaled image saved to: $outputPath")
}
