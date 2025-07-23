"use client";

import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, Download, FileText, Table, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { PDFDocument } from 'pdf-lib';

interface ExtractedData {
    // Company Information
    company_name?: string;
    
    // Vehicle Information
    item_name?: string;
    chassis_number?: string;
    year?: string;
    
    // Vehicle Specifications
    engine_capacity?: string;
    gasoline?: string;
    seat_number?: string;
    weight?: string;
    net_weight?: string;
    length?: string;
    width?: string;
    height?: string;
    
    // Invoice Information
    total_amount?: string;
    pod?: string;
    
    // Contact Information
    email?: string;
    invoice_postal_country?: string;
    invoice_email?: string;
    invoice_phone?: string;
}

interface ProcessingResult {
    success: boolean;
    message: string;
    download_url?: string;
    extracted_data?: ExtractedData;
    error?: string;
}

const PDFProcessor: React.FC = () => {
    const [invoicePdfFile, setInvoicePdfFile] = useState<File | null>(null);
    const [exportCertificatePdfFile, setExportCertificatePdfFile] = useState<File | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<ProcessingResult | null>(null);

    const validatePdfPageCount = async (file: File): Promise<boolean> => {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const pdfDoc = await PDFDocument.load(arrayBuffer);
            if (pdfDoc.getPageCount() > 1) {
                toast.error(`${file.name} has more than one page. Please upload a single-page PDF.`);
                return false;
            }
            return true;
        } catch (error) {
            toast.error(`Failed to read or validate PDF: ${file.name}`);
            return false;
        }
    };

    const handleInvoicePdfFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (file.type === 'application/pdf') {
                if (await validatePdfPageCount(file)) {
                    setInvoicePdfFile(file);
                    setResult(null);
                }
            } else {
                toast.error('Please select a valid PDF file');
            }
        }
    };

    const handleExportCertificatePdfFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (file.type === 'application/pdf') {
                if (await validatePdfPageCount(file)) {
                    setExportCertificatePdfFile(file);
                    setResult(null);
                }
            } else {
                toast.error('Please select a valid PDF file');
            }
        }
    };

    const simulateProgress = () => {
        setProgress(0);
        const totalDuration = 100000; // 100 seconds in milliseconds
        const updateInterval = 1000; // Update every 1 second
        const progressIncrement = (updateInterval / totalDuration) * 100;
        
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 99) {
                    clearInterval(interval);
                    return 99;
                }
                return Math.round(prev + progressIncrement);
            });
        }, updateInterval);
        return interval;
    };

    const handleProcessPDF = async () => {
        if (!invoicePdfFile) {
            toast.error('Please select an invoice PDF file');
            return;
        }

        if (!exportCertificatePdfFile) {
            toast.error('Please select an export certificate PDF file');
            return;
        }

        setIsProcessing(true);
        setResult(null);

        const progressInterval = simulateProgress();

        try {
            const formData = new FormData();
            formData.append('invoice_pdf_file', invoicePdfFile);
            formData.append('export_certificate_pdf_file', exportCertificatePdfFile);

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
            const response = await fetch(`${apiUrl}/api/process-pdf`, {
                method: 'POST',
                body: formData,
            });

            const data: ProcessingResult = await response.json();

            clearInterval(progressInterval);
            setProgress(100);

            setResult(data);

            if (data.success) {
                toast.success('PDF processed successfully!');
            } else {
                toast.error(data.error || 'Processing failed');
            }

        } catch (error) {
            clearInterval(progressInterval);
            setProgress(0);

            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            setResult({
                success: false,
                message: 'Network error',
                error: errorMessage
            });
            toast.error('Failed to process PDF: ' + errorMessage);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDownload = async () => {
        if (!result?.download_url) return;

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
            const downloadUrl = result.download_url.startsWith('http') 
                ? result.download_url 
                : `${apiUrl}${result.download_url}`;
            
            const response = await fetch(downloadUrl);
            if (!response.ok) throw new Error('Download failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `processed_invoice_${invoicePdfFile?.name?.replace('.pdf', '')}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            toast.success('File downloaded successfully!');
        } catch {
            toast.error('Failed to download file');
        }
    };

    return (
        <div className="container mx-auto p-6 max-w-4xl">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900 mb-2">PDF to Excel Processor</h1>
                <p className="text-gray-600">
                    Upload both invoice PDF and export certificate PDF to automatically extract and populate data into our standard Excel template using OCR technology.
                </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                {/* File Upload Section */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Upload className="h-5 w-5" />
                            Upload PDF Files
                        </CardTitle>
                        <CardDescription>
                            Select both PDF files for processing (both are required)
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <Label htmlFor="invoice-pdf-file" className="flex items-center gap-2 mb-2">
                                <FileText className="h-4 w-4" />
                                Invoice PDF (Required)
                            </Label>
                            <Input
                                id="invoice-pdf-file"
                                type="file"
                                accept=".pdf"
                                onChange={handleInvoicePdfFileChange}
                                className="cursor-pointer"
                            />
                            {invoicePdfFile && (
                                <p className="text-sm text-green-600 mt-1">
                                    ✓ {invoicePdfFile.name} ({(invoicePdfFile.size / 1024 / 1024).toFixed(2)} MB)
                                </p>
                            )}
                        </div>

                        <div>
                            <Label htmlFor="export-certificate-pdf-file" className="flex items-center gap-2 mb-2">
                                <FileText className="h-4 w-4" />
                                Export Certificate PDF (Required)
                            </Label>
                            <Input
                                id="export-certificate-pdf-file"
                                type="file"
                                accept=".pdf"
                                onChange={handleExportCertificatePdfFileChange}
                                className="cursor-pointer"
                            />
                            {exportCertificatePdfFile && (
                                <p className="text-sm text-green-600 mt-1">
                                    ✓ {exportCertificatePdfFile.name} ({(exportCertificatePdfFile.size / 1024 / 1024).toFixed(2)} MB)
                                </p>
                            )}
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <Table className="h-4 w-4 text-blue-600" />
                                <span className="text-sm font-medium text-blue-800">Fixed Template</span>
                            </div>
                            <p className="text-sm text-blue-700">
                                Using our standard Excel template. Data from both invoice and export certificate PDFs will be populated at predefined cell positions.
                            </p>
                        </div>

                        <div className="space-y-2">
                            <Button
                                onClick={handleProcessPDF}
                                disabled={!invoicePdfFile || !exportCertificatePdfFile || isProcessing}
                                className="w-full"
                            >
                                {isProcessing ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <Upload className="mr-2 h-4 w-4" />
                                        Process PDFs
                                    </>
                                )}
                            </Button>


                        </div>

                        {isProcessing && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span>Processing...</span>
                                    <span>{progress}%</span>
                                </div>
                                <Progress value={progress} className="w-full" />
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Results Section */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5" />
                            Results
                        </CardTitle>
                        <CardDescription>
                            Processing results and download options
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {result ? (
                            <div className="space-y-4">
                                <Alert className={result.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                                    <div className="flex items-center gap-2">
                                        {result.success ? (
                                            <CheckCircle className="h-4 w-4 text-green-600" />
                                        ) : (
                                            <XCircle className="h-4 w-4 text-red-600" />
                                        )}
                                        <AlertDescription className={result.success ? "text-green-800" : "text-red-800"}>
                                            {result.message}
                                        </AlertDescription>
                                    </div>
                                </Alert>

                                {result.success && result.extracted_data && (
                                    <div className="space-y-3">
                                        <h4 className="font-semibold text-gray-900">Extracted Data:</h4>
                                        <div className="bg-gray-50 p-3 rounded-lg space-y-4">
                                            {(() => {
                                                // Define logical order for displaying fields
                                                const fieldOrder = [
                                                    // Company Information
                                                    'company_name',
                                                    // Vehicle Information
                                                    'item_name',
                                                    'chassis_number',
                                                    'year',
                                                    // Vehicle Specifications
                                                    'engine_capacity',
                                                    'gasoline',
                                                    'seat_number',
                                                    'weight',
                                                    'net_weight',
                                                    'length',
                                                    'width',
                                                    'height',
                                                    // Invoice Information
                                                    'total_amount',
                                                    'pod',
                                                    // Contact Information
                                                    'email',
                                                    'invoice_postal_country',
                                                    'invoice_email',
                                                    'invoice_phone'
                                                ];
                                                
                                                // Create better field labels
                                                const getFieldLabel = (fieldKey: string) => {
                                                    const labels: { [key: string]: string } = {
                                                        'total_amount': 'Total Amount',
                                                        'company_name': 'Company Name',
                                                        'item_name': 'Item Name',
                                                        'chassis_number': 'Chassis Number',
                                                        'gasoline': 'Gasoline',
                                                        'seat_number': 'Seat Number',
                                                        'weight': 'Weight (kg)',
                                                        'engine_capacity': 'Engine Capacity (cc)',
                                                        'net_weight': 'Net Weight (kg)',
                                                        'length': 'Length (cm)',
                                                        'width': 'Width (cm)',
                                                        'height': 'Height (cm)',
                                                        'email': 'Email',
                                                        'pod': 'Port of Destination',
                                                        'invoice_postal_country': 'Invoice Postal/Country',
                                                        'invoice_email': 'Invoice Email',
                                                        'invoice_phone': 'Invoice Phone',
                                                        'year': 'Year'
                                                    };
                                                    return labels[fieldKey] || fieldKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                                };
                                                
                                                // Sort entries according to the defined order
                                                const sortedEntries = Object.entries(result.extracted_data).sort(([a], [b]) => {
                                                    const aIndex = fieldOrder.indexOf(a);
                                                    const bIndex = fieldOrder.indexOf(b);
                                                    // If both are in the order, sort by their position
                                                    if (aIndex !== -1 && bIndex !== -1) {
                                                        return aIndex - bIndex;
                                                    }
                                                    // If only one is in the order, prioritize it
                                                    if (aIndex !== -1) return -1;
                                                    if (bIndex !== -1) return 1;
                                                    // If neither is in the order, sort alphabetically
                                                    return a.localeCompare(b);
                                                });
                                                
                                                // Group fields by sections
                                                const sections = [
                                                    { title: 'Company Information', fields: ['company_name'] },
                                                    { title: 'Vehicle Information', fields: ['item_name', 'chassis_number', 'year'] },
                                                    { title: 'Vehicle Specifications', fields: ['engine_capacity', 'gasoline', 'seat_number', 'weight', 'net_weight', 'length', 'width', 'height'] },
                                                    { title: 'Invoice Information', fields: ['total_amount', 'pod'] },
                                                    { title: 'Contact Information', fields: ['email', 'invoice_postal_country', 'invoice_email', 'invoice_phone'] }
                                                ];
                                                
                                                return sections.map(section => {
                                                    const sectionEntries = sortedEntries.filter(([key]) => section.fields.includes(key));
                                                    if (sectionEntries.length === 0) return null;
                                                    
                                                    return (
                                                        <div key={section.title} className="space-y-2">
                                                            <h5 className="text-sm font-semibold text-gray-700 border-b border-gray-200 pb-1">
                                                                {section.title}
                                                            </h5>
                                                            <div className="space-y-1 pl-2">
                                                                {sectionEntries.map(([key, value]) => (
                                                                    <div key={key} className="flex justify-between">
                                                                        <span className="font-medium text-sm">{getFieldLabel(key)}:</span>
                                                                        <span className="text-gray-700 text-sm">{value || 'N/A'}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    );
                                                });
                                            })()}
                                        </div>
                                    </div>
                                )}

                                {result.success && result.download_url && (
                                    <Button onClick={handleDownload} className="w-full">
                                        <Download className="mr-2 h-4 w-4" />
                                        Download Excel File
                                    </Button>
                                )}

                                {!result.success && result.error && (
                                    <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg">
                                        <strong>Error Details:</strong> {result.error}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-500">
                                <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                                <p>Upload files and click &quot;Process PDFs&quot; to see results here</p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default PDFProcessor;
