from pydantic import BaseModel, Field
from typing import List, Optional

class InvoiceItem(BaseModel):
    description: str = Field(description="Description of the item or service billed")
    quantity: float = Field(default=1.0, description="Quantity billed")
    unit_price: float = Field(description="Unit price per item")
    total_amount: float = Field(description="Total line amount")

class InvoiceSchema(BaseModel):
    vendor_name: str = Field(description="Company or vendor issuing the document")
    invoice_number: Optional[str] = Field(description="Invoice or reference number")
    invoice_date: Optional[str] = Field(description="Date formatted as YYYY-MM-DD")
    line_items: List[InvoiceItem] = Field(description="Itemized line entries")
    subtotal: Optional[float] = Field(description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(description="Stated tax amount")
    grand_total: float = Field(description="Total payable amount")

class ChartAnalysisSchema(BaseModel):
    chart_title: str = Field(description="Title or main subject of the chart")
    chart_type: str = Field(description="Visualization style: Bar, Line, Scatter, Pie, etc.")
    x_axis_label: Optional[str] = Field(description="Metric or variable along the X-axis")
    y_axis_label: Optional[str] = Field(description="Metric or variable along the Y-axis")
    key_observations: List[str] = Field(description="Major numerical trends or high/low points")
    takeaway: str = Field(description="Main conclusion in plain language")