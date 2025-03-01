from fpdf import FPDF
import io

class PDFReport:
    def __init__(self):
        self.pdf = FPDF()

    def generate_report(self, processed_df, forecast_df, metrics, report_type="forecast"):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        
        # Title
        if report_type == "forecast":
            self.pdf.cell(200, 10, txt="Energy Forecast Report", ln=True, align="C")
        elif report_type == "load_balancing":
            self.pdf.cell(200, 10, txt="Load Balancing Report", ln=True, align="C")
        
        # Metrics
        if metrics:
            self.pdf.cell(200, 10, txt=f"MAE: {metrics['mae']:.2f}", ln=True)
            self.pdf.cell(200, 10, txt=f"RMSE: {metrics['rmse']:.2f}", ln=True)
            self.pdf.cell(200, 10, txt=f"MAPE: {metrics['mape']:.2f}%", ln=True)
        
        # Processed Data
        self.pdf.cell(200, 10, txt="Processed Data:", ln=True)
        for index, row in processed_df.iterrows():
            self.pdf.cell(200, 10, txt=str(row.to_list()), ln=True)
        
        # Forecast/Load Balancing Data
        self.pdf.cell(200, 10, txt="Forecast/Load Balancing Data:", ln=True)
        for index, row in forecast_df.iterrows():
            self.pdf.cell(200, 10, txt=str(row.to_list()), ln=True)
        
        
        # Generate PDF in memory
        pdf_bytes = self.pdf.output(dest='S').encode('latin1')
        return io.BytesIO(pdf_bytes)