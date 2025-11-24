import React, { useRef, useEffect } from 'react'

export default function DiseaseBoxesCanvas({ detections = [], imagePreview = null }) {
  const canvasRef = useRef()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')

    if (imagePreview) {
      // Load and draw the uploaded image
      const img = new Image()
      img.onload = () => {
        // Set canvas dimensions to match image
        canvas.width = img.width
        canvas.height = img.height

        // Draw image
        ctx.drawImage(img, 0, 0)

        // Draw classification overlay for ResNet50 (no bounding boxes)
        if (detections && detections.length > 0) {
          // Draw semi-transparent overlay
          ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
          ctx.fillRect(0, 0, canvas.width, canvas.height)

          // Draw detection results at bottom
          detections.forEach((d, i) => {
            const label = `${d.label} (${(d.confidence * 100).toFixed(1)}%)`
            ctx.font = 'bold 16px Arial'
            const metrics = ctx.measureText(label)
            const labelX = (canvas.width - metrics.width) / 2
            const labelY = canvas.height - 30 - i * 30

            ctx.fillStyle = '#ef4444'
            ctx.fillRect(labelX - 8, labelY - 20, metrics.width + 16, 24)

            ctx.fillStyle = '#ffffff'
            ctx.fillText(label, labelX, labelY)
          })
        }
      }
      img.onerror = () => {
        console.error('Failed to load image preview')
      }
      img.src = imagePreview
    } else {
      // Fallback: draw placeholder with gray background
      canvas.width = 420
      canvas.height = 300
      ctx.fillStyle = '#1a2847'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#9aa5b1'
      ctx.font = '14px Arial'
      ctx.fillText('Disease classification — upload image to see results', 8, 20)

      // Draw sample classification result if detections exist
      if (detections && detections.length > 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        detections.forEach((d, i) => {
          const label = `${d.label} (${(d.confidence * 100).toFixed(1)}%)`
          ctx.fillStyle = '#ef4444'
          ctx.font = 'bold 14px Arial'
          const metrics = ctx.measureText(label)
          const labelX = (canvas.width - metrics.width) / 2
          const labelY = canvas.height / 2 + i * 30
          ctx.fillRect(labelX - 8, labelY - 18, metrics.width + 16, 22)
          ctx.fillStyle = '#fff'
          ctx.fillText(label, labelX, labelY)
        })
      }
    }
  }, [detections, imagePreview])

  return (
    <canvas
      ref={canvasRef}
      width={420}
      height={300}
      style={{
        width: '100%',
        borderRadius: 8,
        border: '1px solid var(--border)',
        background: '#1a2847'
      }}
    />
  )
}