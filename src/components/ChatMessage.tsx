import React from 'react';
import { User, Bot } from 'lucide-react';
import type { Message } from '../types';

interface Props {
  message: Message;
}

export const ChatMessage: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';
  
  const formatContent = (content: string) => {
    const sections = content.split('[हिंदी में जवाब]');
    
    return sections.map((section, index) => {
      const lines = section.split('\n').map((line, lineIndex) => {
        if (/^\d+\./.test(line.trim())) {
          return (
            <div key={lineIndex} className="ml-4 my-2 text-base sm:text-lg">
              <strong>{line}</strong>
            </div>
          );
        }
        else if (/^[a-z]\./.test(line.trim())) {
          return (
            <div key={lineIndex} className="ml-8 my-1 text-sm sm:text-base">
              {line}
            </div>
          );
        }
        else if (line.includes('[English Response]')) {
          return (
            <div key={lineIndex} className="font-bold text-base sm:text-lg mb-2 mt-4 text-green-700">
              English Response:
            </div>
          );
        }
        else if (line.includes('[हिंदी में जवाब]')) {
          return (
            <div key={lineIndex} className="font-bold text-base sm:text-lg mb-2 mt-4 text-green-700">
              हिंदी में जवाब:
            </div>
          );
        }
        else if (line.trim()) {
          return <div key={lineIndex} className="my-1 text-sm sm:text-base">{line}</div>;
        }
        return null;
      });

      return (
        <div key={index} className="space-y-1">
          {lines}
        </div>
      );
    });
  };

  return (
    <div className={`flex gap-2 sm:gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? 'bg-green-600' : 'bg-green-700'}`}>
        {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
      </div>
      <div 
        className={`max-w-[85%] rounded-lg p-3 sm:p-4 ${
          isUser 
            ? 'bg-green-600 text-white' 
            : 'bg-white border border-green-100'
        }`}
      >
        {formatContent(message.content)}
      </div>
    </div>
  );
};