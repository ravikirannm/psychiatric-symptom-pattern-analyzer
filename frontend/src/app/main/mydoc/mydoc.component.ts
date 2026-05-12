import { ChangeDetectorRef, Component } from '@angular/core';
import { APIService } from '../../services/api.service';
import { FormControl } from '@angular/forms';
import { ICD11Result, PubmedResult, SymptomAnalysis } from '../../interfaces';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

@Component({
    selector: 'app-mydoc',
    templateUrl: './mydoc.component.html',
    styleUrls: ['./mydoc.component.scss'],
    standalone: false
})
export class MydocComponent {
    progressMessage = '';
    isLoading = false
    queryControl = new FormControl('');
    selectedThread = new FormControl('');
    threads: any[] = []; // This will hold the list of threads fetched from the API
    conversationHistory: any[] = []; // This will 11hold the conversation history for the selected thread
    currentBotMessage = '';
    symptomAnalysis: SymptomAnalysis | null = null;
    icd11Results: ICD11Result[] = [];
    pubmedResults: PubmedResult[] = [];
    finalAnalysis: any = null;
    constructor(private apiService: APIService, private cdr: ChangeDetectorRef, private sanitizer: DomSanitizer) {
        this.fetchMe();
        this.selectedThread.valueChanges.subscribe(threadId => {
            if (threadId) {
                this.fetchThreadData(threadId);
            }
        });
    }

    fetchMe() {
        this.apiService.getData('/me').subscribe({
            next: (response) => {
                console.log('User info from API:', response);
                this.fetchThreads();

                // Handle the response as needed
            }, error: (error) => {
                console.error('Error fetching user info:', error);
            }
        });
    }

    fetchThreads() {
        this.apiService.getData('/threads').subscribe({
            next: (response) => {
                console.log('Threads from API:', response);
                this.threads = response; // Assuming the API returns an array of threads
                this.cdr.markForCheck(); // Trigger change detection to update the UI
            }, error: (error) => {
                console.error('Error fetching threads:', error);
            }
        });
    }

    fetchThreadData(threadId: string) {
        this.apiService.getData(`/thread/${threadId}`).subscribe({
            next: (response) => {
                console.log('Thread data from API:', response);
                this.conversationHistory = [];
                response.forEach((turn: any) => {
                    if (turn.role === 'user' || turn.role === 'assistant') {
                        turn.message = turn.role === 'user' ? turn.content : turn.content.query_response;
                        turn.message = this.getSafeHtml(turn.message);
                        this.conversationHistory.push(turn);
                    }
                    this.fetchAnalysisResults();
                });
                // Handle the response as needed
            }, error: (error) => {
                console.error('Error fetching thread data:', error);
            }
        });
    }

    sendMessage() {
        const query = this.queryControl.value;
        if (query) {
            this.queryControl.setValue(''); // Clear the input field
            let route = '/analyze'
            this.isLoading = true;
            this.conversationHistory.push({
                role: 'user',
                message: query,
                timestamp: new Date()
            });
            const payload:any = { query };
            if (this.selectedThread.value) {
                payload['thread_id'] = this.selectedThread.value;
                
            }
            this.apiService.streamPostData(route, payload).subscribe(
                {
                    next: (event: any) => {
                        if (!event || !event.type) return; // Ignore malformed events
                        if(event.threadId) {
                            this.selectedThread.setValue(event.threadId, { emitEvent: false }); // Update selected thread without triggering fetch
                        }
                        switch (event.type) {
                            case 'progress':
                                this.progressMessage = event.message;
                                break;


                            case 'chat_stream':
                                // Append tokens for the typewriter effect
                                this.currentBotMessage += event.token;
                                // this.currentBotMessage = this.getSafeHtml(this.currentBotMessage) as string; // Sanitize and convert to SafeHtml
                                break;

                            case 'done':
                                this.finalizeMessage();
                                break;
                        }
                        this.cdr.markForCheck(); // Trigger change detection to update the UI
                    },
                    error: (err) => {
                        console.error('Stream failed', err);
                        this.currentBotMessage = 'An error occurred during analysis.';
                        this.finalizeMessage();
                    },
                    complete: () => {
                        if (this.isLoading) this.finalizeMessage();
                    }
                });
        }
    }

    private finalizeMessage() {
        this.isLoading = false;
        if (this.currentBotMessage) {
            this.conversationHistory.push({
                role: 'assistant',
                message: this.getSafeHtml(this.currentBotMessage),
                timestamp: new Date()
            });
            this.currentBotMessage = '';
        }
        this.fetchAnalysisResults()
    }

    fetchAnalysisResults() {
        this.apiService.postData('/fetch-analysis', { thread_id: this.selectedThread.value }).subscribe({
            next: (response) => {
                console.log('Analysis results from API:', response);
                this.finalAnalysis = response;
                this.symptomAnalysis = this.finalAnalysis.symptom_analysis;
                this.icd11Results = this.finalAnalysis?.icd11_results || [];
                this.pubmedResults = this.finalAnalysis?.pubmed_results || [];
                this.cdr.markForCheck(); // Trigger change detection to update the UI
            }
        });
    }

    getSafeHtml(markdownText: string): SafeHtml {
        const rawHtml = marked.parse(markdownText) as string;
        const cleanHtml = DOMPurify.sanitize(rawHtml);
        return this.sanitizer.bypassSecurityTrustHtml(cleanHtml);
    }

    startNewConversation() {
        this.selectedThread.setValue(''); // Clear selected thread
        this.conversationHistory = []; // Clear conversation history
        this.symptomAnalysis = null; // Clear analysis results
        this.icd11Results = [];
        this.pubmedResults = [];
        this.finalAnalysis = null;
    }
}
